#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)

RANK_SIZE=$1
CONFIG_FILE=$2
CASE_NAME=$3
CELL_REUSE=${4:-None}

export RANK_SIZE="$RANK_SIZE"
export MF_PATH=${BASE_PATH}/../../mindformers
export PYTHONPATH=${MF_PATH}:${MF_PATH}/research/deepseek3/:${PYTHONPATH}
export MS_DEV_DUMP_IR_PASSES="step_parallel,validate,hwopt_d_after_inline_graph"
if [ "$CELL_REUSE" = "pp" ]; then
  echo "enable lazy inline in pp"
  export ENABLE_LAZY_INLINE=1
fi
if [ "$CELL_REUSE" = "no_pp" ]; then
  echo "enable lazy inline in no pp"
  export ENABLE_LAZY_INLINE_NO_PIPELINE=1
fi

msrun --worker_num=$RANK_SIZE --local_worker_num=$RANK_SIZE --master_port=8118 --log_dir=$BASE_PATH/$CASE_NAME/ \
  --join=True --cluster_time_out=7200 \
  ${MF_PATH}/run_mindformer.py \
  --config $CONFIG_FILE \
  --register_path ${MF_PATH}/research/deepseek3/
