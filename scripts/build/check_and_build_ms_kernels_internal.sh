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

package_check() {
  absolute_path=$1
  relative_path=${absolute_path#${BASEPATH}/}
  expected_sha256=`git show HEAD:${relative_path} | grep sha256 | awk -F: '{print $NF}'`
  actual_sha256=`sha256sum ${absolute_path} | awk '{print $1}'`
  if [[ "${expected_sha256}" != "${actual_sha256}" ]]; then
    echo "[ERROR] SHA256 hash of ${absolute_path} does not match expected value"
    echo "expected: ${expected_sha256}"
    echo "actual: ${actual_sha256}"
    exit 1
  fi
}

if [[ -n "${MS_INTERNAL_KERNEL_HOME}" ]]; then
  echo "Use local MS_INTERNAL_KERNEL_HOME : ${MS_INTERNAL_KERNEL_HOME}"
  return
fi
if [[ "$(uname)" != Linux || ("$(arch)" != x86_64 && "$(arch)" != aarch64) ]]; then
  echo "[WARNING] Internal kernels only supports linux system, x86_64 or aarch64 CPU arch."
  return
fi
file_path=${BASEPATH}/mindspore/ops/kernel/ascend/internal/prebuild/$(arch)

internal_file_name=${file_path}/ms_kernels_internal.tar.gz
if [[ ! -f "${internal_file_name}" ]]; then
  echo "[WARNING] The file ${internal_file_name}  does NOT EXIST."
  return
fi
internal_file_lines=`cat "${internal_file_name}" | wc -l`
if [[ ${internal_file_lines} -eq 3 ]]; then
  echo "[WARNING] The file ms_kernels_internal.tar.gz is not pulled. Please ensure git-lfs is installed by"
  echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
  return
fi

package_check ${internal_file_name}
tar --warning=no-unknown-keyword -zxf ${internal_file_name} -C ${file_path}
if [[ $? -ne 0 ]]; then
  echo "[WARNING] Unzip ms_kernels_internal.tar.gz FAILED!"
  return
fi
echo "Unzip ms_kernels_internal.tar.gz SUCCESS!"

export MS_INTERNAL_KERNEL_HOME="${file_path}/ms_kernels_internal"
echo "MS_INTERNAL_KERNEL_HOME = ${MS_INTERNAL_KERNEL_HOME}"

dependency_file_name=${file_path}/ms_kernels_dependency.tar.gz
if [[ ! -f "${dependency_file_name}" ]]; then
  echo "[WARNING] The file ${dependency_file_name}  does NOT EXIST."
  return
fi
dependency_file_lines=`cat "${dependency_file_name}" | wc -l`
if [[ ${dependency_file_lines} -eq 3 ]]; then
  echo "[WARNING] The file ms_kernels_dependency.tar.gz is not pulled. Please ensure git-lfs is installed by"
  echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
  return
fi

package_check ${dependency_file_name}
tar --warning=no-unknown-keyword -zxf ${dependency_file_name} -C ${file_path}
if [[ $? -ne 0 ]]; then
  echo "[WARNING] Unzip ms_kernels_dependency.tar.gz FAILED!"
  return
fi

cp -r ${file_path}/ms_kernels_dependency/asdops ${file_path}/ms_kernels_internal
echo "Unzip ms_kernels_dependency.tar.gz SUCCESS!"
