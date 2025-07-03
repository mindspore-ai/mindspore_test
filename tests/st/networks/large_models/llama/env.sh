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
export ASCEND_PATH=/usr/local/Ascend
if [ -d "${ASCEND_PATH}/ascend-toolkit" ]; then
    source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
else
    source ${ASCEND_PATH}/latest/bin/setenv.bash
fi

export DEVICE_MEMORY_CAPACITY=1073741824000
export NOT_FULLY_USE_DEVICES=off
