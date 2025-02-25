#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

export OMP_NUM_THREADS=1

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0  # 1 金山机器
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=1  # https://zhuanlan.zhihu.com/p/653001915

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
echo "Parsed device_ids: ${device_ids[@]}"

stage=5
stop_stage=5

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=$1
rank=$2
echo ${num_nodes} ${rank}


# WenetSpeech training set
set=L
train_set=train_`echo $set | tr 'A-Z' 'a-z'`
dev_set=dev
test_sets="test_net test_meeting"

train_config=conf/run_stage1.yaml
checkpoint=
dir=exp/qwen3
tensorboard_dir=exp/tensorboard
num_workers=8
prefetch=10

decode_checkpoint=${dir}/9.pt
average_checkpoint=false
average_num=5

train_engine=torch_ddp


. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

shards_dir=/nfs-speech-cfs/wangzhou/asr/multimodal/qwenet/data/librispeech/shard
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  for x in train dev test_clean test_other; do
    mkdir -p ${shards_dir}
    tools/make_shard_list_v1.py \
      --resample 16000 \
      --num_utts_per_shard 1000 \
      --num_threads 32 \
      --prefix ${x}\
      data/librispeech/${x}.list\
      ${shards_dir} \
      data/librispeech/${x}_shard.list
  done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training"
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes \
           --nproc_per_node=$num_gpus \
           --node_rank=$rank \
           --master_addr="10.126.203.69" \
           --master_port=54322 \
    wenet/bin/train.py \
      --config $train_config \
      --data_type "shard" \
      --train_data data/librispeech/train_shard.list \
      --cv_data data/librispeech/dev_shard.list \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory 
fi

# test_sets="test_clean test_other"
test_sets="common_voice_v11 voxpopuli_accented voxpopuli tedlium gigaspeech"
test_sets="gigaspeech"
decode_batch=1
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --mode ${average_mode} \
        --max_step ${max_step} \
        --val_best
  fi

  i=0
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}
    mkdir -p ${result_dir}
    device_id=${device_ids[i % ${#device_ids[@]}]}
    echo "Testing ${testset} on GPU ${device_id}"
    python wenet/bin/recognize.py --gpu ${device_id} \
      --config $dir/train.yaml \
      --data_type "raw" \
      --test_data data/librispeech/${testset}.list \
      --checkpoint $decode_checkpoint \
      --batch_size ${decode_batch} \
      --result_dir $result_dir  \
      --gpu ${device_id} \
      --device "cuda" \
      &
    ((i++))
    if [[ $device_id -eq $((num_gpus - 1)) ]]; then
      wait
    fi
  }
  done
  wait
fi