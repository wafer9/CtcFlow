# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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

from typing import List, Optional
import logging
import os

import time
import urllib
import hmac
import hashlib
import base64
import requests
import json


from wenet.utils.common import (StepTimer, lrs_to_str, tensor_to_scalar)


def log_per_step(writer, info_dict, timer: Optional[StepTimer] = None):
    tag = info_dict["tag"]
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    epoch = info_dict.get('epoch', 0)
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1) if tag != "CV" else 1
    log_interval = info_dict.get('log_interval', 10)
    lrs = info_dict.get("lrs", [0.0])
    is_gradient_accumulation_boundary = info_dict.get(
        "is_gradient_accumulation_boundary", False)

    rank = int(os.environ.get('RANK', 0))
    # TRAIN Tensorboard
    if tag == "TRAIN" and rank == 0 and writer is not None:
        if (train_engine == "deepspeed" and is_gradient_accumulation_boundary
            ) or (train_engine in ["torch_ddp", "torch_fsdp"] and
                  (batch_idx + 1) % accum_grad == 0):
            writer.add_scalar('train/train_loss',
                              tensor_to_scalar(loss_dict['loss']), step)
            writer.add_scalar('train/grad_norm', info_dict['grad_norm'], step)
            for name, value in loss_dict.items():
                if name != 'loss' and value is not None:
                    writer.add_scalar('train/{}'.format(name),
                                      tensor_to_scalar(value), step)
            # lr
            for i, lr in enumerate(lrs):
                writer.add_scalar('train/lr_{}'.format(i), lr, step)
    # CV Tensorboard
    elif "CV" in tag and rank == 0 and writer is not None:
        for name, value in loss_dict.items():
            writer.add_scalar('cv/{}'.format(name), tensor_to_scalar(value), step)
        return

    # TRAIN & CV, Shell log (stdout)
    if (batch_idx + 1) % log_interval == 0:
        log_str = '{} | '.format(tag)
        if timer is not None:
            timer_step = step
            if info_dict.get("cv_step", None) is not None:
                timer_step = info_dict['cv_step']
            steps_per_second = timer.steps_per_second(timer_step)
            log_str += 'steps/sec {:.3f}| '.format(steps_per_second)
        log_str += 'Batch {}/{} loss {:.6f} '.format(
            epoch, batch_idx + 1 if 'save_interval' not in info_dict else
            (step + 1), tensor_to_scalar(loss_dict['loss']))
        for name, value in loss_dict.items():
            if name != 'loss' and value is not None:
                log_str += '{} {:.6f} '.format(name, tensor_to_scalar(value))
        if tag == "TRAIN":
            log_str += 'lr {} grad_norm {:.6f} rank {}'.format(
                lrs_to_str(lrs), info_dict['grad_norm'], rank)
            log_str += ' bs {}'.format(info_dict['bs'])
        logging.debug(log_str)



def send_dingtalk(mode, epoch, step, loss, lr):
    headers = {'Content-Type': 'application/json', "Charset": "UTF-8"}
    prefix = 'https://oapi.dingtalk.com/robot/send?access_token=85dc3415b8a0ab868c1ae2bbb5914f3636d422231d8bf79ef2c51f096fdb94b0' # finetune
    prefix = 'https://oapi.dingtalk.com/robot/send?access_token=ed129e0db150a3f538463e8336f4d042334d1cd1296dbf42a84d8f89cd73bc67'
    timestamp = str(round(time.time() * 1000))
    secret = 'secret...'
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    url = f'{prefix}&timestamp={timestamp}&sign={sign}'
    data = {
        "at": {
            "isAtAll": False
        },
        "text": {
            "content": f'mode: {mode}, epoch: {epoch}, step: {step}, loss: {loss:.3f} lr: {lr:.9f}'
        },
        "msgtype": "text"
    }

    requests.post(url=url, data=json.dumps(data), headers=headers)
