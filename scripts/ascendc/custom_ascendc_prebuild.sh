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

if [[ "$(uname)" != Linux || ("$(arch)" != x86_64 && "$(arch)" != aarch64) ]]; then
  echo "[WARNING] Custom Ascend C only supports linux system, x86_64 or aarch64."
  return
fi
file_path=${BASEPATH}/mindspore/ops/kernel/ascend/ascendc/prebuild/$(arch)
ascendc_file_name=${file_path}/prebuild_ascendc.tar.gz
if [[ ! -f "${ascendc_file_name}" ]]; then
  echo "[WARNING] The file ${ascendc_file_name}  does NOT EXIST."
  return
fi
ascendc_file_lines=$(cat "${ascendc_file_name}" | wc -l)
if [[ ${ascendc_file_lines} -eq 3 ]]; then
  echo "[WARNING] The file prebuild_ascendc.tar.gz is not pulled. Please ensure git-lfs is installed by"
  echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
  return
fi
tar --warning=no-unknown-keyword -zxf ${ascendc_file_name} -C ${file_path}
if [[ $? -ne 0 ]]; then
  echo "[WARNING] Unzip prebuild_ascendc.tar.gz FAILED!"
  return
fi
echo "Unzip prebuild_ascendc.tar.gz SUCCESS!"
