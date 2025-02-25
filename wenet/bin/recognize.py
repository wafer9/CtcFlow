# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader
from wenet.dataset.dataset import Dataset

from wenet.text.tokenization_qwen_audio import QWenAudioTokenizer
from wenet.transformer.modeling_lam import LAMModel
from wenet.utils.checkpoint import load_checkpoint
from whisper.normalizers import EnglishTextNormalizer

from wenet.utils.wer import compute_wer
from peft import LoraConfig, get_peft_model
normalizer = EnglishTextNormalizer()
from wenet.utils.textnorm_zh import textnorm_zh
import time
import re


def post_process(sentence: str):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\u4e00-\u9fffa-z\s\']', ' ', sentence)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    chars = pattern.split(sentence)
    mix_chars = [w.strip() for w in chars if len(w.strip()) > 0]
    tokens = []
    for ch_or_w in mix_chars:
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        else:
            tokens.extend(ch_or_w.split())
    sentence = " ".join(tokens)
    return sentence



def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--lang',
                        type=str,
                        default='en',
                        choices=['zh', 'en', 'mix'],
                        help='model\'s dtype')
    
    parser.add_argument('--prompt',
                        type=str,
                        default='<|wo_itn|>',
                        choices=['<|wo_itn|>', '<|itn|>', ""],
                        help='model\'s dtype')

    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    test_conf['is_inference'] = True
    test_conf['prompt'] = args.prompt

    if configs['text_config']['model_type'] == 'qwen2':
        from wenet.text.tokenization_qwen2_audio import Qwen2AudioTokenizer
        tokenizer_config = configs['tokenizer']
        tokenizer = Qwen2AudioTokenizer.from_pretrained(
                    configs['text_model'],
                    **tokenizer_config)
        tokenizer.add_special_tokens(151936)
    else:
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

    if configs['lora_text_model']:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model.text_model = get_peft_model(model.text_model, lora_config)
        model.text_model.print_trainable_parameters()
    infos = load_checkpoint(model, args.checkpoint)
    if configs['lora_text_model']:
        lm_add = torch.cat([model.text_model.base_model.model.base_model.embed_tokens.weight.data, model.lm_head_add.weight.data], dim=0)
        emd_add = torch.cat([model.text_model.base_model.model.base_model.embed_tokens.weight.data, model.wte_add.weight.data], dim=0)
        model.text_model.lm_head.weight.data = lm_add
        model.text_model.base_model.model.base_model.embed_tokens.weight.data = emd_add
    else:
        lm_add = torch.cat([model.text_model.model.embed_tokens.weight.data, model.lm_head_add.weight.data], dim=0)
        emd_add = torch.cat([model.text_model.model.embed_tokens.weight.data, model.wte_add.weight.data], dim=0)
        model.text_model.lm_head.weight.data = lm_add
        model.text_model.model.embed_tokens.weight.data = emd_add


    test_dataset = Dataset(args.data_type,
                         args.test_data,
                         tokenizer,
                         test_conf,
                         partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    # Init asr model from configs
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    generate_config = {
        'do_sample': False,
        'max_new_tokens': 256,
        'min_new_tokens': None,
        'length_penalty': 1.0,
        'num_return_sequences': 1,
        'repetition_penalty': 1.0,
        'output_hidden_states': True,
    }

    os.makedirs(args.result_dir, exist_ok=True)
    file_name = os.path.join(args.result_dir, 'text')
    file_p = open(file_name, 'w')

    H, D, S, I = 0, 0, 0, 0
    audio_durtion, real_time = 0, time.time()
    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(test_data_loader):
                keys = batch_dict["keys"]
                input_features=batch_dict['feats'].to(device)
                input_ids=batch_dict['input_ids'].to(device)
                audio_attention_mask=batch_dict['audio_attention_mask'].to(device)
                adapter_out_lengths = batch_dict['adapter_out_lengths'].to(device)
                input_wavs = batch_dict['wavs'].to(device)
                raw_audio_attention_mask=batch_dict['raw_audio_attention_mask'].to(device)

                output_ids = model.generate(
                    input_features=input_features, 
                    input_ids=input_ids, 
                    audio_attention_mask=audio_attention_mask,
                    adapter_out_lengths=adapter_out_lengths,
                    audio_info=batch_dict['audio_info'],
                    input_wavs=input_wavs,
                    raw_audio_attention_mask=raw_audio_attention_mask,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generate_config
                )
                # eos_token_id_tensor = torch.tensor([tokenizer.pad_token_id]).to(
                # output_ids.device).unsqueeze(0).repeat(output_ids.size(0), 1)
                # # output_ids = torch.cat([output_ids, eos_token_id_tensor], dim=1)
                # eos_token_pos = output_ids.eq(tokenizer.pad_token_id).float().argmax(-1)
                # print("12: ", output_ids.eq(tokenizer.pad_token_id).float())

                for i in range(output_ids.size(0)):
                    pred_id = output_ids[i].tolist()
                    hyp = tokenizer.decode(pred_id, skip_special_tokens=True, 
                                                audio_info=batch_dict['audio_info'])
                    ref = batch_dict['labels'][i]
                    if args.lang == 'en':
                        ref_tn = normalizer(post_process(ref))
                        hyp_tn = normalizer(post_process(hyp))
                    elif args.lang == 'zh':
                        # ref_tn = textnorm_zh(post_process(ref))
                        # hyp_tn = textnorm_zh(post_process(hyp))
                        ref_tn = textnorm_zh(ref)
                        hyp_tn = textnorm_zh(hyp)
                    elif args.lang == 'mix':
                        ref_tn = post_process(ref)
                        hyp_tn = post_process(hyp)
                    measures = compute_wer(ref_tn, hyp_tn, lan=args.lang)
                    line = '\n{}\n ref: {}\n hyp: {}\n ref_tn: {}\n hyp_tn: {}\n H:{} D:{} S:{} I:{} wer:{:.2f}\n'.format(
                                keys[i], ref, hyp, ref_tn, hyp_tn,
                                measures['hits'], measures['deletions'], measures['substitutions'], measures['insertions'],
                                measures['wer'])
                    H += measures['hits']
                    D += measures['deletions']
                    S += measures['substitutions']
                    I += measures['insertions']
                    audio_durtion += input_features[i].size(0)/100.0
                    logging.info(line)
                    file_p.write(line + '\n')
                    file_p.flush()

    file_p.write('H:{} D:{} S:{} I:{} WER:{}\n'.format(H, D, S, I, (S+D+I)/(S+D+H)))
    real_time = time.time() - real_time
    file_p.write('{} {} RTF: {}\n'.format(audio_durtion, real_time, real_time/audio_durtion))
    file_p.close()



if __name__ == '__main__':
    main()
