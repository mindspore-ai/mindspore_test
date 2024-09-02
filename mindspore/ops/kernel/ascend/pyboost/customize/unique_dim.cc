/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/pyboost/customize/unique_dim.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr> UniqueDimAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor, const BoolImmPtr &sorted,
  const BoolImmPtr &return_inverse, const Int64ImmPtr &dim) {
  MS_LOG(DEBUG) << "Run device task unique_dim start";
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  OpRunner::InferOpOutput(op, input_tensor, sorted, return_inverse, dim);

  auto sorted_imm = GetValue<bool>(sorted);
  auto return_inverse_imm = GetValue<bool>(return_inverse);
  auto dim_imm = GetValue<int64_t>(dim);

  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, op->outputs());

  runtime::Pipeline::Get().WaitForward();

  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);
  // Run sync
  auto return_value =
    LAUNCH_ACLNN_SYNC(aclnnUniqueDim, device_context, stream_id, input_tensor, sorted_imm, return_inverse_imm, dim_imm,
                      outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  const auto &cache_func_ptr = std::get<2>(return_value);
  auto all_acl_tensor = cache_func_ptr(false, {}, true);

  // update shape
  auto output_real_shape0 = all_acl_tensor[4];
  auto output_real_shape1 = all_acl_tensor[5];
  auto output_real_shape2 = all_acl_tensor[6];
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape0, output_real_shape1, output_real_shape2};
  op->UpdateOutputShape(op->output(kIndex0), output_real_shape0);
  op->UpdateOutputShape(op->output(kIndex1), output_real_shape1);
  op->UpdateOutputShape(op->output(kIndex2), output_real_shape2);
  MS_LOG(DEBUG) << "Run device task unique_dim end";

  return std::make_tuple(op->output(kIndex0), op->output(kIndex1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
