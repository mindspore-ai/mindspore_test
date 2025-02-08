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

#include "kernel/ascend/pyboost/customize/reshape.h"
#include "mindspore/ccsrc/pyboost/customize/reshape.h"
#include "mindspore/ops/view/reshape_strides_calc.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                             const ValueTuplePtr &shape) {
  auto old_storage_info = input_tensor->storage_info();
  if (old_storage_info != nullptr && !old_storage_info->is_contiguous) {
    auto primitive = op->primitive();
    auto storage_info_list = ops::ReshapeCalc(primitive, {input_tensor, shape});
    if (!storage_info_list.empty()) {
      MS_LOG(DEBUG) << "View Uncontiguous Reshape Call start";
      tensor::BaseTensorPtrList outputs;
      PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
      PyBoostUtils::CreateOutputTensor(op->device_context(), input_tensor, storage_info_list, &outputs);

      op->set_outputs(outputs);
      PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
        MS_LOG(DEBUG) << "View device task Uncontiguous Reshape start";
        auto device_context = op->device_context();
        PyBoostUtils::MallocOpInputs(device_context, input_tensor);
        MS_LOG(DEBUG) << "View device task Uncontiguous Reshape end";
      }));

      MS_LOG(DEBUG) << "View Uncontiguous Reshape Call end";
      return op->output(0);
    }
  }

  MS_LOG(DEBUG) << "View Reshape Call start";
  return ReshapeCustomize(op, input_tensor, shape, op->device_context()->device_context_key_.device_name_);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
