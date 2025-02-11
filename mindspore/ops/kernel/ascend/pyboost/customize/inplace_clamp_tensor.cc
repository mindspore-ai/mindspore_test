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

#include "kernel/ascend/pyboost/customize/inplace_clamp_tensor.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplaceClampTensorAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const BaseTensorPtr &input_tensor,
                                                        const std::optional<BaseTensorPtr> &min_tensor,
                                                        const std::optional<BaseTensorPtr> &max_tensor) {
  MS_LOG(DEBUG) << "Call InplaceClampTensor start";
  OpRunner::InferOpOutput(op, input_tensor, min_tensor, max_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, min_tensor, max_tensor);

  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, min_tensor, max_tensor]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, min_tensor, max_tensor);

    // Inplace output need be front
    LAUNCH_ACLNN(aclnnClampTensor, device_context, op->stream_id(), input_tensor, min_tensor, max_tensor, input_tensor);
    MS_LOG(DEBUG) << "Launch InplaceClampTensor end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
