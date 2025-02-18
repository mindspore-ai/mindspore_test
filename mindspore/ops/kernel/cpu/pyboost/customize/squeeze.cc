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

#include "kernel/cpu/pyboost/customize/squeeze.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/cpu/pyboost/pyboost_cpu_custom_kernel_register.h"
#include "mindspore/ops/view/squeeze_strides_calc.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
void SqueezeCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                         const std::optional<ValueTuplePtr> &dim) {
  MS_LOG(DEBUG) << "View Squeeze Call start";
  TensorStorageInfoPtrList storage_info_list;
  if (!dim.has_value()) {
    storage_info_list = ops::SqueezeCalc(op->primitive(), {input_tensor, mindspore::kNone});
  } else {
    storage_info_list = ops::SqueezeCalc(op->primitive(), {input_tensor, dim.value()});
  }
  if (!storage_info_list.empty()) {
    tensor::BaseTensorPtrList outputs;
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
    PyBoostUtils::CreateOutputTensor(op->device_context(), input_tensor, storage_info_list, &outputs);
    op->set_outputs(outputs);
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
      MS_LOG(DEBUG) << "View device task Squeeze start";
      PyBoostUtils::MallocOpInputsForView(op->device_context(), input_tensor);
      MS_LOG(DEBUG) << "View device task Squeeze end";
    }));
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << op->primitive() << " or input ERROR";
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
