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

#include "kernel/ascend/pyboost/customize/masked_select.h"

#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr MaskedSelectAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                              const TensorPtr &mask_tensor) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  OpRunner::InferOpOutput(op, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, op->outputs());
  runtime::Pipeline::Get().WaitForward();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor, mask_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);
  const auto &all_acl_tensor =
    LAUNCH_ACLNN_SYNC(aclnnMaskedSelect, device_context, op->stream_id(), input_tensor, mask_tensor, outputs[0]);

  auto output_real_shape = all_acl_tensor[kIndex2];
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape};

  op->UpdateOutputShape(op->output(kIndex0), output_real_shape);
  MS_LOG(DEBUG) << "Run device task MaskedSelect end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
