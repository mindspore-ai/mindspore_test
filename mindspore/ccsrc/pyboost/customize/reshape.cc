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

#include "mindspore/ccsrc/pyboost/customize/reshape.h"
#include <string>
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/pyboost/auto_generate/copy.h"
#include "mindspore/ccsrc/pyboost/auto_generate/view.h"
#include "mindspore/ops/view/view_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ReshapeCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                   const ValueTuplePtr &shape_ptr, const std::string &device_target) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto shape = GetValue<std::vector<int64_t>>(shape_ptr);
  auto view_op = CREATE_PYBOOST_OP(View, device_target);
  auto old_storage_info = input_tensor->storage_info();
  if (old_storage_info == nullptr || old_storage_info->is_contiguous) {
    // Tensor is contiguous, reshape by view
    auto output_tensor = view_op->Call(input_tensor, shape);
    op->set_outputs(view_op->outputs());
    return output_tensor;
  }

  // Tensor is not contiguous, need call copy first
  auto copy_op = CREATE_PYBOOST_OP(Copy, device_target);
  copy_op->set_stream_id(op->stream_id());
  const auto copy_tensor = copy_op->Call(input_tensor);
  auto output_tensor = view_op->Call(copy_tensor, shape);
  op->set_outputs(view_op->outputs());
  return output_tensor;
}

tensor::TensorPtr ReshapeCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                   const std::vector<int64_t> &shape, const std::string &device_target) {
  MS_LOG(DEBUG) << "Call start";
  auto old_storage_info = input_tensor->storage_info();
  TensorStorageInfoPtrList storage_info_list;
  TensorPtr real_tensor;
  if (old_storage_info == nullptr || old_storage_info->is_contiguous) {
    // Tensor is contiguous, reshape by view
    storage_info_list = ops::ViewCalcImpl(input_tensor, shape);
    real_tensor = input_tensor;
  } else {
    // Tensor is not contiguous, need call copy first
    auto copy_op = CREATE_PYBOOST_OP(Copy, op->device_context()->device_context_key_.device_name_);
    copy_op->set_stream_id(op->stream_id());
    const auto copy_tensor = copy_op->Call(input_tensor);
    real_tensor = copy_tensor;
    storage_info_list = ops::ViewCalcImpl(copy_tensor, shape);
  }
  if (!storage_info_list.empty()) {
    tensor::TensorPtrList outputs;
    // Create device address for input tensors
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), real_tensor);
    PyBoostUtils::CreateOutputTensor(op->device_context(), real_tensor, storage_info_list, &outputs);
    // Async
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, real_tensor]() {
      MS_LOG(DEBUG) << "View device task View start";
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputsForView(device_context, real_tensor);
      MS_LOG(DEBUG) << "View device task View end";
    }));
    op->set_outputs(outputs);
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << op->primitive()->name() << " or input ERROR";
  }
  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
