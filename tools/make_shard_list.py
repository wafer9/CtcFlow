#!/usr/bin/env python3

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

import argparse
import io
import logging
import os
import tarfile
import time
import multiprocessing
import json

import torch
import torchaudio
torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def write_tar_file(data_list,
                   tar_file,
                   resample=16000,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        prev_wav = None
        for item in data_list:
            key, wav, text, label = item

            suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS

            # read & resample
            ts = time.time()
            if isinstance(wav, str):
                with open(wav, 'rb') as f:
                    wav_file = f.read()
            with io.BytesIO(wav_file) as file_obj:
                audio, sample_rate = torchaudio.load(file_obj)
            audio = torchaudio.transforms.Resample(sample_rate,
                                                    resample)(audio)
            read_time += (time.time() - ts)

            audio = (audio * (1 << 15))
            audio = audio.to(torch.int16)
            ts = time.time()
            with io.BytesIO() as f:
                torchaudio.save(f,
                                audio,
                                resample,
                                format="wav",
                                bits_per_sample=16)
                suffix = "wav"
                f.seek(0)
                data = f.read()
            save_time += (time.time() - ts)

            assert isinstance(text, str)
            ts = time.time()
            txt_file = key + '.text'
            text = text.encode('utf8')
            text_data = io.BytesIO(text)
            text_info = tarfile.TarInfo(txt_file)
            text_info.size = len(text)
            tar.addfile(text_info, text_data)

            label_file = key + '.label'
            label = label.encode('utf8')
            label_data = io.BytesIO(label)
            label_info = tarfile.TarInfo(label_file)
            label_info.size = len(label)
            tar.addfile(label_info, label_data)

            wav_file = key + '.' + suffix
            wav_data = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            tar.addfile(wav_info, wav_data)
            write_time += (time.time() - ts)
        logging.info('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--resample',
                        type=int,
                        default=16000,
                        help='segments file')
    parser.add_argument('list_file', help='list file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)

    data = []
    with open(args.list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = json.loads(line.strip())
            key, wav, text, label = arr['key'], arr['wav'], arr['text'], arr['label']
            # label = label.replace(' ', '').replace('▁', ' ')
            data.append((key, wav, text, label))

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(
            write_tar_file,
            (chunk, tar_file, args.resample, i, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
