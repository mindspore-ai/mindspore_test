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

#include "pipeline/llm_boost/utils.h"
#include "mindapi/base/format.h"
#include "include/common/utils/utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/device_address.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace pipeline {
tensor::TensorPtr SetFormat(const py::object &py_tensor, const std::string &format_name) {
  auto tensor = IsStubTensor(py_tensor) ? ConvertStubTensor(py_tensor) : py_tensor.cast<tensor::TensorPtr>();
  if (format_name != kOpFormat_ND && format_name != kOpFormat_FRAC_NZ) {
    MS_LOG(ERROR) << "The format " << format_name
                  << " is not supported. The format only supports 'ND' and 'FRACTAL_NZ'";
    return tensor;
  }
  if (tensor->DataDim() <= 1) {
    MS_LOG(DEBUG) << "The dimension of tensor is less than or equal to 1, and not need to convert the format";
    return tensor;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_name = device_context->device_context_key().device_name_;
  auto device_id = device_context->device_context_key().device_id_;
  auto stream_id = device_context->device_res_manager_->DefaultStream();
  auto device_sync = tensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  if (device_address != nullptr) {
    device_address->set_format(format_name);
    return tensor;
  }

  mindspore::Format format = mindspore::Format::ND;
  if (format_name == kOpFormat_FRAC_NZ) {
    format = mindspore::Format::FRACTAL_NZ;
  }
  device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, static_cast<size_t>(tensor->data().nbytes()), tensor->shape(), format, tensor->data_type(), device_name,
    device_id, stream_id);
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_from_persistent_mem(tensor->is_parameter());
  tensor->set_device_address(device_address);
  return tensor;
}
}  // namespace pipeline
}  // namespace mindspore
