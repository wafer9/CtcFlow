# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import logging
import os
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from wenet.utils.common import StepTimer

from wenet.utils.train_utils import log_per_step
from torch.nn.utils import clip_grad_norm_
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint

class Executor:

    def __init__(self,
                 global_step: int = 0,
                 device: torch.device = torch.device("cpu")):
        self.step = global_step + 1
        self.train_step_timer = None
        self.cv_step_timer = None
        self.device = device

    def train(self, model, optimizer, scheduler, train_data_loader,
              writer, configs, rank):
        ''' Train one epoch
        '''
        if self.train_step_timer is None:
            self.train_step_timer = StepTimer(self.step)
        accum_grad = configs.get('accum_grad', 50.0)
        clip = configs.get('grad_clip', 50.0)
        log_interval = configs.get('log_interval', 10)
        model.train()
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx

                input_features=batch_dict['feats'].to(self.device)
                input_ids=batch_dict['input_ids'].to(self.device)
                audio_attention_mask=batch_dict['audio_attention_mask'].to(self.device)
                encoder_out_lengths = batch_dict['encoder_out_lengths'].to(self.device)
                input_wavs = batch_dict['wavs'].to(self.device)
                raw_audio_attention_mask=batch_dict['raw_audio_attention_mask'].to(self.device)
                labels=batch_dict['labels'].to(self.device)
                info_dict["bs"] = labels.size(0)

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict.get("train_engine", "torch_ddp") in [
                        "torch_ddp", "torch_fsdp"
                ] and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    loss_dict = model(input_features=input_features, 
                                 input_ids=input_ids, 
                                 audio_attention_mask=audio_attention_mask,
                                 encoder_out_lengths=encoder_out_lengths,
                                 labels=labels,
                                 input_wavs=input_wavs,
                                 raw_audio_attention_mask=raw_audio_attention_mask,
                                 audio_info=batch_dict['audio_info'])
                    info_dict['loss_dict'] = loss_dict
                    loss = info_dict['loss_dict']['loss']
                loss.backward()
                grad_norm = 0.0
                if (batch_idx + 1) % accum_grad == 0:
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    optimizer.zero_grad()
                    grad_norm = grad_norm.item()
                scheduler.step()
                self.step += 1
                info_dict["lrs"] = [x['lr'] for x in optimizer.param_groups]
                info_dict["grad_norm"] = grad_norm

                # write training: tensorboard && log
                log_per_step(writer, info_dict, timer=self.train_step_timer)
                self.step += 1 if (batch_idx +
                                   1) % info_dict["accum_grad"] == 0 else 0
                if self.step % 1000 == 0 and rank == 0:
                    save_model_path = os.path.join(configs['model_dir'], '{}_{}.pt'.format(configs['epoch'], self.step))
                    save_checkpoint(
                        model,
                        save_model_path,
                        {
                            'epoch': configs['epoch'],
                            'cv_loss': loss.item(),
                            'step': self.step,
                            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        }
                    )

    def cv(self, model, cv_data_loader, writer, configs):
        ''' Cross validation on
        '''
        if self.cv_step_timer is None:
            self.cv_step_timer = StepTimer(0.0)
        else:
            self.cv_step_timer.last_iteration = 0.0
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_loss = 1, {}, 0  # avoid division by 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict["cv_step"] = batch_idx

                input_features = batch_dict['feats'].to(self.device)
                input_ids = batch_dict['input_ids'].to(self.device)
                audio_attention_mask = batch_dict['audio_attention_mask'].to(self.device)
                encoder_out_lengths = batch_dict['encoder_out_lengths'].to(self.device)
                input_wavs = batch_dict['wavs'].to(self.device)
                raw_audio_attention_mask=batch_dict['raw_audio_attention_mask'].to(self.device)
                labels = batch_dict['labels'].to(self.device)

                num_utts = batch_dict["labels"].size(0)
                num_seen_utts += num_utts
                if num_utts == 0:
                    continue
                loss_dict = model(input_features=input_features, 
                                 input_ids=input_ids, 
                                 audio_attention_mask=audio_attention_mask,
                                 encoder_out_lengths=encoder_out_lengths,
                                 labels=labels,
                                 input_wavs=input_wavs,
                                 raw_audio_attention_mask=raw_audio_attention_mask,
                                 audio_info=batch_dict['audio_info'])
                total_loss += loss_dict['loss'].item() * num_utts
                info_dict['loss_dict'] = loss_dict

        # write cv: log
        log_per_step(writer, info_dict, timer=self.cv_step_timer)
        
        return total_loss/num_seen_utts
