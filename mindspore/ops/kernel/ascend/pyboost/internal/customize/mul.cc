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

#include "kernel/ascend/pyboost/internal/customize/mul.h"
#include "kernel/ascend/pyboost/internal/functions/functions.h"

#include "runtime/pynative/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/auto_generate/contiguous.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::BaseTensorPtr TensorContiguous(const tensor::BaseTensorPtr &tensor) {
  if (tensor == nullptr || tensor->storage_info() == nullptr) {
    return tensor;
  }
  const auto &old_device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(old_device_address);

  const DeviceContext *device_context = runtime::OpRunner::GetDeviceContext(old_device_address->device_name());
  MS_EXCEPTION_IF_NULL(device_context);
  GilReleaseWithCheck release_gil;
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key().device_name_);
  const auto &contiguous_tensor = contiguous_op->Call(tensor);
  return contiguous_tensor;
}
}  // namespace

void InternalMulAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                const BaseTensorPtr &other_tensor) {
  OpRunner::InferOpOutput(op, input_tensor, other_tensor);

  // internal op does not support noncontiguous tensor
  auto contiguous_input = TensorContiguous(input_tensor);
  auto contiguous_other = TensorContiguous(other_tensor);
  auto contiguous_output = TensorContiguous(op->outputs()[kIndex0]);
  op->set_outputs({contiguous_output});

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), contiguous_input, contiguous_other);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  internal_mul(op, contiguous_input, contiguous_other);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
