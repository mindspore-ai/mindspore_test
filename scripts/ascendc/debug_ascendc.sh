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

script_path=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ms_path=$(realpath "${script_path}/../../")
echo "The ms project is located in: ${ms_path}"
ascendc_910_path=${ms_path}/mindspore/ops/kernel/ascend/ascendc/custom_ascendc_910
ascendc_910b_path=${ms_path}/mindspore/ops/kernel/ascend/ascendc/custom_ascendc_910b
ascendc_ascendc_ops_path=${ms_path}/mindspore/ops/kernel/ascend/ascendc/prebuild/$(arch)/custom_ascendc_ops/
build_build_path=${ms_path}/build/package/build/lib/mindspore/lib/plugin/ascend
build_ms_path=${ms_path}/build/package/mindspore/lib/plugin/ascend
ms_python_path=${ms_path}/mindspore/python/mindspore/lib/plugin/ascend
echo "Custom ascendc prebuild ops path: ${ascendc_ascendc_ops_path}"
echo "Package build path: ${build_build_path}"
echo "Package ms path: ${build_ms_path}"

copy_ascendc_custom() {
  echo "Copy ascendc prebuild ops to build dictionary."
  cp -r ${ascendc_ascendc_ops_path}/* ${build_build_path}
  cp -r ${ascendc_ascendc_ops_path}/* ${build_ms_path}
  cp -r ${ascendc_ascendc_ops_path}/* ${ms_python_path}
}

delete_ascendc_custom() {
  echo "Delete custom ascendc cache."
  rm -rf ${ascendc_910_path} ${ascendc_910b_path} ${ascendc_ascendc_ops_path} ${ms_python_path}/custom_ascendc_*
  rm -rf ${build_build_path}/custom_ascendc_* ${build_ms_path}/custom_ascendc_*
}

if [[ "$1" != "" ]]; then
  case $1 in
  -c)
    copy_ascendc_custom
    ;;
  -d)
    delete_ascendc_custom
    ;;
  -f)
    delete_ascendc_custom
    bash ${script_path}/ascendc_compile.sh
    if [ $? -eq 0 ]; then
      copy_ascendc_custom
    fi
    ;;
  esac
fi
