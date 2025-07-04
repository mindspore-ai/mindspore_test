/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/pyboost/customize/new_zeros.h"
#include "kernel/ascend/pyboost/auto_generate/zeros.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void NewZerosAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                             const ValueTuplePtr &size, const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "NewZeros Call start";

  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;

  std::optional<Int64ImmPtr> use_dtype;
  if (dtype.has_value()) {
    use_dtype.emplace(dtype.value());
  } else {
    auto tensor_type = input_tensor->Dtype()->type_id();
    use_dtype.emplace(std::make_shared<Int64Imm>(tensor_type));
  }

  auto zeros_op = CREATE_PYBOOST_OP(Zeros, device_name);
  auto zeros_out = zeros_op->Call(size, use_dtype);
  op->set_outputs({zeros_out});

  MS_LOG(DEBUG) << "NewZeros Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
