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

#include "kernel/ascend/pyboost/customize/inplace_masked_scatter.h"
#include <memory>
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplaceMaskedScatterAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                          const BaseTensorPtr &input, const BaseTensorPtr &mask,
                                                          const BaseTensorPtr &source) {
  auto input_type_id = input->data_type();
  if (input_type_id == kNumberTypeFloat64 || input_type_id == kNumberTypeInt16) {
    MS_EXCEPTION(ValueError) << "For InplaceMaskedScatter, the type of 'input' is no support Tensor[Float64, Int16] ";
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, mask, source);
  op->set_outputs({input});
  // source
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, mask, source]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input, mask, source);
    // Inplace output need be front
    MS_LOG(DEBUG) << "Call InplaceMaskedScatter start";
    LAUNCH_ACLNN(aclnnInplaceMaskedScatter, device_context, op->stream_id(), input, mask, source);
    MS_LOG(DEBUG) << "Launch InplaceMaskedScatter end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
