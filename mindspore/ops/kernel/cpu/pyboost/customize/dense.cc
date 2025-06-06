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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dense.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/transpose.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/contiguous.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/matmul_ext.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<int64_t> GetTransposePerm(const TensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  size_t size = shape.size();
  std::vector<int64_t> perm(size);
  if (size < kDim2) {
    perm[0] = 0;
    return perm;
  }
  perm[size - kDim1] = static_cast<int64_t>(size - kDim2);
  perm[size - kDim2] = static_cast<int64_t>(size - kDim1);
  for (size_t i = 0; i < size - kDim2; ++i) {
    perm[i] = static_cast<int64_t>(i);
  }
  return perm;
}
}  // namespace

void DenseCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                       const TensorPtr &weight_tensor, const std::optional<TensorPtr> &bias_tensor) {
  MS_LOG(DEBUG) << "Dense Launch start";
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, bias_tensor);
  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;
  auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_name);
  auto perm = GetTransposePerm(weight_tensor);
  auto matmul_op = CREATE_PYBOOST_OP(MatMulExt, device_name);

  auto output = matmul_op->Call(input_tensor, contiguous_op->Call(transpose_op->Call(weight_tensor, perm)));

  if (bias_tensor.has_value()) {
    auto add_op = CREATE_PYBOOST_OP(Add, device_name);
    output = add_op->Call(output, bias_tensor.value());
  }
  op->set_outputs({output});
  MS_LOG(DEBUG) << "Dense Launch end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
