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

#include "mindspore/ops/kernel/cpu/pyboost/customize/to_device.h"
#include "pynative/utils/pyboost/customize/to.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ToDeviceCPUCustomize(const std::shared_ptr<OpRunner> &op,
                                       const mindspore::tensor::TensorPtr &self_tensor,
                                       const std::optional<mindspore::Int64ImmPtr> &device,
                                       const std::optional<mindspore::Int64ImmPtr> &dtype,
                                       const mindspore::BoolImmPtr &non_blocking, const mindspore::BoolImmPtr &copy) {
  return ToDeviceCustomize(op, self_tensor, device, dtype, non_blocking, copy);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
