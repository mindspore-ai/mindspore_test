#!/bin/bash

export DEVICE_NUM=8
export RANK_SIZE=8
export MS_ENABLE_GE=1

file_name=$1
cache_path=$2
rank_table_path=$4
export RANK_TABLE_FILE=$rank_table_path
export GLOG_v=2
export MS_COMPILER_CACHE_ENABLE=1
export MS_COMPILER_CACHE_PATH=$cache_path

msrun --worker_num=8 --local_worker_num=8 $file_name
