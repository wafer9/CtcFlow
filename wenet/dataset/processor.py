# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
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

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from langid.langid import LanguageIdentifier, model
import logging
import librosa
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from wenet.text.tokenization_qwen_audio import QWenAudioTokenizer
import whisper
import re
from typing import List, Dict, Union
from wenet.utils.mask import make_pad_mask


torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

logging.getLogger('langid').setLevel(logging.INFO)


def get_T_after_pool(L_in, model_type='whisper', dilation=1):
    if model_type == "whisper":
        conv = "[(1,3,2)] + [(1,3,2)]"
    else:
        conv = "[(1,3,1)] + [(1,3,2)] + [(4,8,8)]"    
    for (padding, kernel_size, stride) in eval(conv):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out

def get_T_data2vec(L_in, dilation=1):

    feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
    def _conv_out_length(input_length, kernel_size, stride):
        return torch.floor((input_length - kernel_size) / stride + 1)

    for i in range(len(feature_enc_layers)):
        L_in = _conv_out_length(
            L_in,
            feature_enc_layers[i][1],
            feature_enc_layers[i][2],
        )
    # L_in = torch.floor((L_in + 1) / 2) # TimeReductionLayer1D, ×2
    l_out1 = L_in.to(torch.int32)
    conv = '[(4,8,8)]'         # ×4
    for (padding, kernel_size, stride) in eval(conv):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return l_out1, L_out.to(torch.int32)

import os
try:
    cpu_info = os.popen("lscpu | grep 'Vendor ID'").read()
    # 0x48 --> HiSilicon
    if (cpu_info.rstrip().split(" ")[-1] == "0x48"):
        # NOTE (MengqingCao): set number of threads in the subprocesses to 1
        # Why? There may be some operators ultilizing multi-threads in processor,
        # causing possibly deadlock in Kunpeng.
        # Similar issue in PyTorch: https://github.com/pytorch/pytorch/issues/45198
        torch.set_num_threads(1)
except Exception as ex:
    logging.warning('Failed to set number of thread in Kunpeng, \
        this may cause segmentfault while dataloading, \
        ignore this warning if you are not using Kunpeng')


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    # sample['wav'] = sample['audio']
    # del sample['audio']
    assert 'wav' in sample
    obj = json.loads(sample['text'])
    sample['text'] = obj['text']
    sample['label'] = obj['label']

    wav_file = sample['wav']
    if isinstance(wav_file, str):
        with open(wav_file, 'rb') as f:
            wav_file = f.read()
    if 'start' in sample:
        assert 'end' in sample
        sample_rate = torchaudio.info(wav_file).sample_rate
        start_frame = int(sample['start'] * sample_rate)
        end_frame = int(sample['end'] * sample_rate)
        with io.BytesIO(wav_file) as file_obj:
            waveform, _ = torchaudio.load(file_obj,
                                          num_frames=end_frame - start_frame,
                                          frame_offset=start_frame)
    else:
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    sample['wav_file'] = obj['audio']
    del sample['wav']
    sample['wav'] = waveform  # overwrite wav
    sample['sample_rate'] = sample_rate
    return sample


def singal_channel(sample, channel=0):
    """ Choose a channel of sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            channel: target channel index

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'wav' in sample
    waveform = sample['wav']
    channel_nums = waveform.size(0)
    assert channel < channel_nums
    if channel_nums != 1:
        waveform = waveform[channel, :].unsqueeze(0)
    sample['wav'] = waveform
    return sample


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav

    return sample


def compute_whisper_fbank(sample, n_mels=80):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    assert sample_rate == 16000
    try:
        mat = whisper.log_mel_spectrogram(sample['wav'], n_mels=n_mels)
    except:
        print("sample['wav']: ", sample['wav'].shape, sample['key'])
    sample['feat'] = mat[0].transpose(0,1)

    with torch.no_grad():
        wav = sample['wav'][0]
        wav_ln = F.layer_norm(wav, wav.shape)
    sample['wav_ln'] = wav_ln.unsqueeze(1)
    return sample

# def compute_raw(sample):
#     """ Extract fbank

#         Args:
#             sample: {key, wav, sample_rate, ...}

#         Returns:
#             {key, feat, wav, sample_rate, ...}
#     """
#     assert 'sample_rate' in sample
#     assert 'wav' in sample
#     assert 'key' in sample
#     sample_rate = sample['sample_rate']
#     assert sample_rate == 16000
#     with torch.no_grad():
#         feats = sample['wav'][0]
#         feats = F.layer_norm(feats, feats.shape)
#     sample['feat'] = feats.unsqueeze(1)
#     return sample


def filter(sample,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {key, wav, label, sample_rate, ...}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            bool: True to keep, False to filter
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False

    if 'Emilia/ZH' not in sample['wav_file']:
        return False

    if 'label' in sample:
        if len(sample['label']) < token_min_length:
            return False
        if len(sample['label']) > token_max_length:
            return False
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                return False
            if len(sample['label']) / num_frames > max_output_input_ratio:
                return False
    return True


def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            sample: {key, feat, ...}
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            {key, feat, ....}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            sample: Iterable{key, feat, ...}
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, ...}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


def sort_by_feats(sample):
    assert 'feat' in sample
    assert isinstance(sample['feat'], torch.Tensor)
    return sample['feat'].size(0)

def tokenize(sample, tokenizer: QWenAudioTokenizer, 
             is_inference=False,
             prompt=""):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    audio_pattern=rf"{tokenizer.audio_start_tag}(.*?){tokenizer.audio_end_tag}"
    audio_pad_tag = tokenizer.audio_pad_tag
    def _prepare_text(texts: List[str],
                      length_features: torch.Tensor) -> List[str]:
        new_texts = []
        audio_index = 0
        for text in texts:
            new_text, pre_end = "", 0
            matches = re.finditer(audio_pattern, text)
            if matches is not None:
                for match in matches:
                    new_text += text[pre_end:match.start(1)]
                    new_text += audio_pad_tag * length_features[
                        audio_index]
                    pre_end = match.end(1)
                    audio_index += 1
            new_text += text[pre_end:]
            new_texts.append(new_text)
        assert audio_index == length_features.shape[0]
        return new_texts
    
    encoder_out_lengths = get_T_after_pool(sample['feats_lengths'], model_type='whisper')
    adapter_out_lengths = get_T_after_pool(sample['feats_lengths'], model_type='whisper_down')
    
    text = [_ + prompt for _ in sample['texts']]
    text = _prepare_text(text, adapter_out_lengths)
    label = [_ + tokenizer.eos_token for _ in sample['labels']]

    if is_inference:
        encoding = tokenizer(text=text,
                            text_target=label,
                            padding=True,
                            return_tensors="pt",
                            return_attention_mask=True)
    else:
        encoding = tokenizer(
                            text=text,
                            text_pair=label,
                            add_special_tokens=True,
                            padding=True,  # truncation
                            return_tensors="pt",
                            return_token_type_ids=True,
                            return_attention_mask=True)
        token_type_ids = encoding["token_type_ids"].clone()
        sample['labels'] = encoding["input_ids"].clone()
        sample["labels"][token_type_ids == 0] = -100

    sample['input_ids'] = encoding["input_ids"].clone()

    bos_pos = torch.where(encoding["input_ids"] == tokenizer.audio_start_id)
    eos_pos = torch.where(encoding["input_ids"] == tokenizer.audio_end_id)
    assert (bos_pos[0] == eos_pos[0]).all()
    audio_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
    sample['audio_info'] = {"audio_pos": audio_pos}

    sample['encoder_out_lengths'] = encoder_out_lengths
    sample['adapter_out_lengths'] = adapter_out_lengths
    sample['audio_attention_mask'] = ~make_pad_mask(sample['feats_lengths'])
    sample['raw_audio_attention_mask'] = ~make_pad_mask(sample['wavs_lengths'])
        
    return sample



def padding(data,):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    wavs = [sample[i]['wav_ln'] for i in order]
    feats = [sample[i]['feat'] for i in order]
    keys = [sample[i]['key'] for i in order]
    texts = [sample[i]['text'] for i in order]
    labels = [sample[i]['label'] for i in order]
    padded_feats = pad_sequence(feats,
                                batch_first=True,
                                padding_value=0)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    padded_wavs = pad_sequence(wavs,
                                batch_first=True,
                                padding_value=0)
    wavs_lengths = torch.tensor([sample[i]['wav_ln'].size(0) for i in order],
                                 dtype=torch.int32)

    batch = {
        "keys": keys,
        "feats": padded_feats,
        "texts": texts,
        "labels": labels,
        "feats_lengths": feats_lengths,
        "wavs":padded_wavs,
        "wavs_lengths":wavs_lengths
    }
    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
