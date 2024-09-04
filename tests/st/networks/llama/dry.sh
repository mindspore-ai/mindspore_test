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
TEST_MODE=$2
export MS_SIMULATION_LEVEL=1
export RANK_SIZE=4
export RANK_ID=0
export GLOG_v=1
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign"
export ENABLE_LAZY_INLINE_NO_PIPELINE=$1

python ${BASE_PATH}/test_dryrun_llama.py --test_mode ${TEST_MODE} > ${TEST_MODE}.log 2>&1