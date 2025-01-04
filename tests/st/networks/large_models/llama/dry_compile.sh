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
PARALLEL_MODE=$1
TEST_MODE=$2
export MS_SIMULATION_LEVEL=1
export GLOG_v=2
export RANK_SIZE=32
export RANK_ID=0
export MS_DEV_RUNTIME_CONF="compile_statistics:True"
export MS_ENBALE_NUMA=1

if [ "$PARALLEL_MODE" = "semi" ]; then
  python ${BASE_PATH}/test_dryrun_llama_semi_compile.py --test_mode ${TEST_MODE} > ${TEST_MODE}.log 2>&1
elif [ "$PARALLEL_MODE" = "auto" ]; then
  python ${BASE_PATH}/test_dryrun_llama_auto_compile.py --test_mode ${TEST_MODE} > ${TEST_MODE}_auto.log 2>&1
fi