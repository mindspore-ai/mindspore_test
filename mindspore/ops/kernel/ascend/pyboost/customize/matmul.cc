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

#include "kernel/ascend/pyboost/customize/matmul.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/pyboost/auto_generate/transpose.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace matmul {
ValueTuplePtr GetTransposePerm(const TensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  size_t size = shape.size();
  std::vector<ValuePtr> perm(size);
  if (size < kDim2) {
    auto zero = std::make_shared<Int64Imm>(0);
    perm[0] = MakeValue(zero);
    return std::make_shared<ValueTuple>(perm);
  }
  perm[size - kDim1] = MakeValue(static_cast<int64_t>(size - kDim2));
  perm[size - kDim2] = MakeValue(static_cast<int64_t>(size - kDim1));
  for (size_t i = 0; i < size - kDim2; ++i) {
    perm[i] = MakeValue(static_cast<int64_t>(i));
  }
  return std::make_shared<ValueTuple>(perm);
}
}  // namespace matmul
tensor::TensorPtr MatMulAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                        const TensorPtr &mat2_tensor, const BoolImmPtr &transpose_a,
                                        const BoolImmPtr &transpose_b) {
  OpRunner::InferOpOutput(op, input_tensor, mat2_tensor, transpose_a, transpose_b);
  auto transpose_a_imm = GetValue<bool>(transpose_a);
  auto transpose_b_imm = GetValue<bool>(transpose_b);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mat2_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  TensorPtr input_tensor_ = input_tensor;
  if (transpose_a_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    input_tensor_ = transpose_op->Call(input_tensor, matmul::GetTransposePerm(input_tensor));
  }

  TensorPtr mat2_tensor_ = mat2_tensor;
  if (transpose_b_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    mat2_tensor_ = transpose_op->Call(mat2_tensor, matmul::GetTransposePerm(mat2_tensor));
  }
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor_, mat2_tensor_]() {
    MS_LOG(DEBUG) << "Run device task MatMul start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor_, mat2_tensor_);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
    auto cube_math_type = GetCubeMathType(IsAllowMatmulHF32());
    LAUNCH_ACLNN(aclnnMm, device_context, op->stream_id(), input_tensor_, mat2_tensor_, outputs[0], cube_math_type);
    MS_LOG(DEBUG) << "Run device task MatMul end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
