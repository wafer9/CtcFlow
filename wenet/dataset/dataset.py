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

from functools import partial
import sys
from typing import Optional
from wenet.dataset import processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)

from wenet.utils.file_utils import read_symbol_table


def Dataset(data_type,
            data_list_file,
            tokenizer,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    # cycle dataset
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    dataset = dataset.map_ignore_error(processor.decode_wav)

    singal_channel_conf = conf.get('singal_channel_conf', {})
    dataset = dataset.map(
        partial(processor.singal_channel, **singal_channel_conf))

    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = dataset.map(partial(processor.speed_perturb))

    fbank_conf = conf.get('fbank_conf', {})
    dataset = dataset.map(partial(processor.compute_whisper_fbank, 
                                n_mels=fbank_conf['num_mel_bins']))

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))

    # spec_aug = conf.get('spec_aug', True)
    # spec_sub = conf.get('spec_sub', False)
    # if spec_aug:
    #     spec_aug_conf = conf.get('spec_aug_conf', {})
    #     dataset = dataset.map(partial(processor.spec_aug, **spec_aug_conf))
    # if spec_sub:
    #     spec_sub_conf = conf.get('spec_sub_conf', {})
    #     dataset = dataset.map(partial(processor.spec_sub, **spec_sub_conf))


    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=processor.padding)
    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=processor.padding,
        )
    
    is_inference = conf.get('is_inference', True)
    prompt = conf.get('prompt', "")
    dataset = dataset.map(partial(processor.tokenize, 
                                  tokenizer=tokenizer, 
                                  is_inference=is_inference,
                                  prompt=prompt))

    return dataset
