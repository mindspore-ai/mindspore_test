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

#include "kernel/ascend/pyboost/customize/inplace_elu.h"
#include <memory>
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceEluAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                            const ScalarPtr &alpha) {
  MS_LOG(DEBUG) << "Call InplaceElu start";
  TypeId data_type = input_tensor->data_type();
  if (data_type == kNumberTypeFloat64) {
    MS_LOG(EXCEPTION) << "Unsupported input dtype: float64, because aclnnEluBackward does not support dtype: float64";
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, alpha]() {
    MS_LOG(DEBUG) << "Run device task InplaceElu start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);

    static const ScalarPtr scale = std::make_shared<FP32Imm>(1.f);
    static const ScalarPtr input_scale = std::make_shared<FP32Imm>(1.f);
    LAUNCH_ACLNN(aclnnInplaceElu, device_context, op->stream_id(), input_tensor, alpha, scale, input_scale);
    MS_LOG(DEBUG) << "Run device task InplaceElu end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
