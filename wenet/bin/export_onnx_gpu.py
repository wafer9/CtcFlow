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
import onnxruntime
from pathlib import Path


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


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

    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
    print(args)
    return args


def export_whisper_encoder(audio_model, 
                           input_features, 
                           audio_attention_mask,
                           encoder_outpath):
    audio_model.forward = audio_model.infer
    input_features = input_features.transpose(1, 2) # (B, T, 80) -> (B, 80, T)
    t1 = time.time()
    audios = audio_model(input_features, audio_attention_mask)
    print(audios.sum(), time.time() - t1)

    inputs = (input_features, audio_attention_mask)
    dynamic_axes = {
            'input_features': {0: 'B', 2: 'T'},
            'audio_attention_mask': {0: 'B', 1: 'T'},
            'whisper_out': {0: 'B', 1: 'T1'},
        }
    print('save onnx model: %s' % encoder_outpath)
    outpath = Path(encoder_outpath).parent
    outpath.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
            audio_model, inputs, encoder_outpath, opset_version=14,
            export_params=True, do_constant_folding=True,
            input_names=['input_features', 'audio_attention_mask'],
            output_names=['whisper_out'],
            dynamic_axes=dynamic_axes, 
            verbose=False)
    
    # onnx_encoder = onnx.load(encoder_outpath)
    providers=['CUDAExecutionProvider'] # or 'TensorrtExecutionProvider'
    ort_session = onnxruntime.InferenceSession(encoder_outpath, providers=providers)
    ort_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    ort_inputs = {
                'input_features': to_numpy(input_features),
                'audio_attention_mask': to_numpy(audio_attention_mask)
            }
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_encoder_out = torch.tensor(ort_outs[0])
    print(onnx_encoder_out[0].sum(), time.time() - t1)
    

def export_llm(asr_model, args):
    pass


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
    infos = load_checkpoint(model, args.checkpoint)

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

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    generate_config = {
        'do_sample': False,
        'max_new_tokens': 256,
        'min_new_tokens': None,
        'length_penalty': 1.0,
        'num_return_sequences': 1,
        'repetition_penalty': 1.0,
        'output_hidden_states': True,
    }

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_data_loader):
            keys = batch_dict["keys"]
            input_features=batch_dict['feats'].to(device)
            input_ids=batch_dict['input_ids'].to(device)
            audio_attention_mask=batch_dict['audio_attention_mask'].to(device)
            adapter_out_lengths = batch_dict['adapter_out_lengths'].to(device)
            input_wavs = batch_dict['wavs'].to(device)
            raw_audio_attention_mask=batch_dict['raw_audio_attention_mask'].to(device)

            export_whisper_encoder(model.audio_model, 
                                   input_features, 
                                   audio_attention_mask,
                                   encoder_outpath=os.path.join(args.result_dir, 'whisper.onnx'))
            # export_llm(model.text_model)


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
            eos_token_id_tensor = torch.tensor([tokenizer.pad_token_id]).to(
            output_ids.device).unsqueeze(0).repeat(output_ids.size(0), 1)
            output_ids = torch.cat([output_ids, eos_token_id_tensor], dim=1)
            eos_token_pos = output_ids.eq(tokenizer.pad_token_id).float().argmax(-1)
            for i in range(output_ids.size(0)):
                pred_id = output_ids[i, :eos_token_pos[i]].tolist()
                hyp = tokenizer.decode(pred_id, skip_special_tokens=True, 
                                            audio_info=batch_dict['audio_info'])
                xx = 1
            break












if __name__ == "__main__":
    main()
