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

#include "kernel/ascend/pyboost/customize/weight_quant_batch_matmul.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/pyboost/auto_generate/transpose.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "ir/device_address_maker.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void WeightQuantBatchMatmulV2AscendCall(const std::shared_ptr<OpRunner> &op,
                                        const device::DeviceContext *device_context, const TensorPtr &x_tensor,
                                        const TensorPtr &weight_tensor, const TensorPtr &antiquant_scale_tensor,
                                        const std::optional<TensorPtr> &antiquant_offset_tensor,
                                        const std::optional<TensorPtr> &quant_scale_tensor,
                                        const std::optional<TensorPtr> &quant_offset_tensor,
                                        const std::optional<TensorPtr> &bias_tensor, int64_t antiquant_group_size,
                                        const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  LAUNCH_ACLNN(aclnnWeightQuantBatchMatmulV2, device_context, op->stream_id(), x_tensor, weight_tensor,
               antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor,
               antiquant_group_size, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
}
std::vector<int64_t> GetWeightQuantBatchMatmulPerm(const TensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  int64_t size = shape.size();
  std::vector<int64_t> perm(size);
  if (size < ops::kSize2) {
    perm[0] = 0;
    return perm;
  }
  perm[size - 1] = size - SizeToLong(kDim2);
  perm[size - kDim2] = size - 1;
  for (int64_t i = 0; i < size - ops::kSize2; ++i) {
    perm[i] = i;
  }
  return perm;
}
}  // namespace
tensor::TensorPtr WeightQuantBatchMatmulV2AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, const TensorPtr &weight_tensor,
  const TensorPtr &antiquant_scale_tensor, const std::optional<TensorPtr> &antiquant_offset_tensor,
  const std::optional<TensorPtr> &quant_scale_tensor, const std::optional<TensorPtr> &quant_offset_tensor,
  const std::optional<TensorPtr> &bias_tensor, const BoolImmPtr &transpose_x, const BoolImmPtr &transpose_weight,
  const Int64ImmPtr &antiquant_group_size) {
  OpRunner::InferOpOutput(op, x_tensor, weight_tensor, antiquant_scale_tensor, antiquant_offset_tensor,
                          quant_scale_tensor, quant_offset_tensor, bias_tensor, transpose_x, transpose_weight,
                          antiquant_group_size);
  auto transpose_x_imm = GetValue<bool>(transpose_x);
  auto transpose_weight_imm = GetValue<bool>(transpose_weight);
  auto antiquant_group_size_imm = GetValue<int64_t>(antiquant_group_size);

  TensorPtr new_weight_tensor = weight_tensor;
  auto tensor_type = op->input_abs()[kIndex1]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  if (tensor_type->element()->type_id() == kNumberTypeInt4) {
    ShapeVector weight_shape = weight_tensor->shape();
    int kInt4ShapeMul = 2;
    weight_shape.back() *= kInt4ShapeMul;
    const ShapeVector &new_weight_shape = weight_shape;
    new_weight_tensor = std::make_shared<tensor::Tensor>(weight_tensor->data_type(), new_weight_shape,
                                                         weight_tensor->data_c(), weight_tensor->DataNBytes());
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, new_weight_tensor,
                                antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
                                quant_offset_tensor, bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  TensorPtr x_tensor_trans = x_tensor;
  if (transpose_x_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    x_tensor_trans = transpose_op->Call(x_tensor_trans, GetWeightQuantBatchMatmulPerm(x_tensor_trans));
  }
  TensorPtr weight_tensor_trans = new_weight_tensor;
  if (transpose_weight_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    weight_tensor_trans = transpose_op->Call(weight_tensor_trans, GetWeightQuantBatchMatmulPerm(weight_tensor_trans));
  }
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor_trans, weight_tensor_trans, antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
     quant_offset_tensor, bias_tensor, antiquant_group_size_imm]() {
      MS_LOG(DEBUG) << "Run device task weight quant batchMatmul v2 start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor_trans, weight_tensor_trans, antiquant_scale_tensor,
                                   antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      WeightQuantBatchMatmulV2AscendCall(op, device_context, x_tensor_trans, weight_tensor_trans,
                                         antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
                                         quant_offset_tensor, bias_tensor, antiquant_group_size_imm, outputs);
      MS_LOG(DEBUG) << "Run device task weight quant batchMatmul v2 end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
