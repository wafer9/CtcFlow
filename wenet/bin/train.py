# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import datetime
import logging
import os
import torch
import yaml
import copy

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record
from wenet.utils.common import lrs_to_str, TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu

from wenet.utils.executor import Executor
from wenet.utils.config import override_config


from wenet.text.tokenization_qwen_audio import QWenAudioTokenizer
from wenet.transformer.modeling_lam import LAMModel
from transformers import AutoFeatureExtractor

from wenet.dataset.dataset import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from wenet.utils.scheduler import WarmupLR
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from glob import glob
from wenet.utils.train_utils import send_dingtalk

from peft import LoraConfig, get_peft_model


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=2,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo', "hccl"],
                        help='distributed backend')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
    return args


# NOTE(xcsong): On worker errors, this recod tool will summarize the
#   details of the error (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    seed=777
    generator = torch.Generator()
    generator.manual_seed(seed)


    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # init tokenizer
    tokenizer = QWenAudioTokenizer.from_pretrained(
                configs['tokenizer']
                if configs['tokenizer'] is not None else configs['text_model'],
                trust_remote_code=True,
                use_fast=configs['use_fast_tokenizer'])

    model = LAMModel.from_audio_text_pretrained(
            configs['audio_model'], 
            configs['text_model'],
            configs['audio_config'], 
            configs['text_config'],
            configs['num_special_tokens_add'],
            tokenizer)
    
    if glob(os.path.join(args.model_dir+'/*.pt')):
        checkpoints = glob(os.path.join(args.model_dir+'/*.pt'))
        try:
            infos = load_checkpoint(model, sorted(checkpoints)[-1])
        except:
            infos = {}
    else:
        infos = {}

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    # if model.ctc_model is not None:
    #     _freeze_params(model.ctc_model)

    if configs['freeze_audio_model']:
        _freeze_params(model.audio_model)
        # _freeze_params(model.proj)


    train_conf = configs['dataset_conf']
    train_conf['is_inference'] = False
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['shuffle'] = False


    train_dataset = Dataset(args.data_type, 
                            args.train_data, 
                            tokenizer,
                            train_conf, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         tokenizer,
                         cv_conf,
                         partition=False)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   persistent_workers=True,
                                   generator=generator,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                generator=generator,
                                prefetch_factor=args.prefetch)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.dist_backend)

    writer = None
    if rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {:,d}'.format(num_params))

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('the number of model requires_grad params: {:,d}'.format(num_params))

        # Writer
        os.makedirs(args.model_dir, exist_ok=True)
        exp_id = os.path.basename(args.model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)


    optimizer = optim.Adam(model.parameters(), lr=configs['optim_conf']['lr'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True)
    device = torch.device("cuda")
    device = int(os.environ.get('LOCAL_RANK', 0))

    executor = Executor(device=device)

    step = infos.get('step', -1)
    executor.step = step
    scheduler.set_step(step)

    start_epoch = max(0, infos.get('epoch', -1) + 1)
    num_epochs = configs.get('max_epoch', 100)

    for epoch in range(start_epoch, num_epochs):
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        # dist.barrier() # Ensure all ranks start Train at the same time.
        configs['model_dir'] = args.model_dir
        executor.train(model, optimizer, scheduler, train_data_loader, writer, configs, rank)
        # dist.barrier() # Ensure all ranks start CV at the same time.
        loss = executor.cv(model, cv_data_loader, writer, configs)

        if rank == 0:
            logging.info('Epoch {} CV info lr {} cv_loss {}'.format(epoch, lr, loss))
            save_model_path = os.path.join(args.model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model,
                save_model_path, 
                {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': loss,
                    'step': executor.step,
                    'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                }
            )
            send_dingtalk(configs['run_name'], epoch, executor.step, loss, lr)


if __name__ == '__main__':
    main()
