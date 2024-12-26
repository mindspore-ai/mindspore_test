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
CONFIG_FILE=$1
TEST_MODE=$2
MODEL_NAME=$3
SCRIPT_NAME=$4
USE_DEVICE_NUM=$5
PORT=$6
VALID_NPUS=$7

export ASCEND_RT_VISIBLE_DEVICES=${VALID_NPUS}

msrun --worker_num ${USE_DEVICE_NUM} --local_worker_num ${USE_DEVICE_NUM} --log_dir ${BASE_PATH}/${TEST_MODE} \
 --master_port ${PORT} --join True \
 ${BASE_PATH}/${MODEL_NAME}/${SCRIPT_NAME} \
 --yaml_file ${CONFIG_FILE} \
 --test_mode ${TEST_MODE} > ${BASE_PATH}/${TEST_MODE}.log 2>&1
