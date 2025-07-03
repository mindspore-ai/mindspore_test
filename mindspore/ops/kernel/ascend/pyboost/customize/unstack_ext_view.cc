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

#include "kernel/ascend/pyboost/customize/unstack_ext_view.h"

#include "mindspore/ops/view/unstack_ext_view_strides_calc.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/customize/op_common.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "runtime/device/device_address_utils.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> UnstackExtViewAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const TensorPtr &x_tensor, const int64_t &axis) {
  MS_LOG(DEBUG) << "View UnstackExt Call start";
  auto primitive = op->primitive();
  auto storage_info_list = ops::UnstackExtViewBasicTypeCalc(x_tensor, axis);
  if (!storage_info_list.empty()) {
    std::vector<tensor::TensorPtr> outputs;
    // Create device address for input tensors
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
    PyBoostUtils::CreateOutputTensor(op->device_context(), x_tensor, storage_info_list, &outputs);

    op->set_outputs(outputs);

    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
      MS_LOG(DEBUG) << "View device task UnstackExt start";
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, x_tensor);
      MS_LOG(DEBUG) << "View device task UnstackExt end";
    }));
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive->name() << " or input ERROR";
  }
  MS_LOG(DEBUG) << "View UnstackExt Call end";
  return op->outputs();
}

std::vector<tensor::TensorPtr> UnstackExtViewAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const TensorPtr &x_tensor, const Int64ImmPtr &dim) {
  MS_LOG(DEBUG) << "View UnstackExt Call start";
  auto primitive = op->primitive();
  auto storage_info_list = ops::UnstackExtViewCalc(primitive, {x_tensor, dim});
  if (!storage_info_list.empty()) {
    std::vector<tensor::TensorPtr> outputs;
    // Create device address for input tensors
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
    PyBoostUtils::CreateOutputTensor(op->device_context(), x_tensor, storage_info_list, &outputs);

    op->set_outputs(outputs);

    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
      MS_LOG(DEBUG) << "View device task UnstackExtView start";
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, x_tensor);
      MS_LOG(DEBUG) << "View device task UnstackExtView end";
    }));
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive->name() << " or input ERROR";
  }
  MS_LOG(DEBUG) << "View UnstackExtView Call end";
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
