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

#include "kernel/ascend/pyboost/customize/inplace_masked_fill_tensor.h"

#include <memory>

#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceMaskedFillTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                         const TensorPtr &mask, const TensorPtr &value) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, mask, value);
  op->set_outputs({input});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, mask, value]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input, mask, value);
    // Inplace output need be front
    MS_LOG(DEBUG) << "Call MaskFillTensor start";
    LAUNCH_ACLNN(aclnnInplaceMaskedFillTensor, device_context, op->stream_id(), input, mask, value);
    MS_LOG(DEBUG) << "Launch MaskFillTensor end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
