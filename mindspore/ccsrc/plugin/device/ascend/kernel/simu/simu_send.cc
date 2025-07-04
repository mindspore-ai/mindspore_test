/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/ascend/kernel/simu/simu_send.h"

namespace mindspore {
namespace kernel {
bool SimuSendKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Invalid simu send input size (" << inputs.size() << ").";
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
