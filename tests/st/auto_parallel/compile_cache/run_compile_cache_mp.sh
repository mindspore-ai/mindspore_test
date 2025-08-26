#!/bin/bash

file_name=$1
cache_path=$2
log_dir_root=$3
export GLOG_v=2
export MS_COMPILER_CACHE_ENABLE=1
export MS_COMPILER_CACHE_PATH=$cache_path

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10811 --join=True \
--log_dir=$log_dir_root pytest -s $file_name
