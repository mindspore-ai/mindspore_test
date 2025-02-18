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
#include "ir/tensor.h"
#include "utils/ms_context.h"
#include "include/common/utils/tensor_py.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/log_adapter.h"
#include "mindapi/base/format.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// Reuse src tensor's device address by dst tensor. For internal usage only.
void ReuseDataPtr(const tensor::TensorPyPtr &dst_, const tensor::TensorPyPtr &src_, size_t offset) {
  // get context meta
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // get device
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_ctx);
  device_ctx->Initialize();

  auto stream_id = device_ctx->device_res_manager_->DefaultStream();

  // create src device address if null
  auto src = src_->GetTensor();
  if (src->device_address() == nullptr) {
    auto device_ptr = device_ctx->device_res_manager_->AllocateMemory(src->Size(), stream_id);
    auto src_device_address = device_ctx->device_res_manager_->CreateDeviceAddress(
      reinterpret_cast<void *>(device_ptr), src->Size(), src->shape(), Format::DEFAULT_FORMAT, src->data_type(),
      device_name, device_id, stream_id);

    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(device_ptr) << ", size:" << src->Size()
                  << ", shape:" << src->shape() << ", data_type:" << TypeIdToString(src->data_type());
    MS_EXCEPTION_IF_NULL(src_device_address);

    src_device_address->SyncHostToDevice(src->Size(), src->data_c());
    src->set_device_address(src_device_address);
  }

  // create device address with src ptr
  uint8_t *ptr = reinterpret_cast<uint8_t *>(src->device_address()->GetMutablePtr());
  auto dst = dst_->GetTensor();
  auto offset_size = offset * UnitSizeInBytes(dst->data_type());
  if (offset_size >= src->Size()) {
    MS_EXCEPTION(ValueError) << "Offset overflow. Expect offset in bytes less than " << src->Size() << ", got "
                             << offset_size;
  }

  auto dst_device_address = device_ctx->device_res_manager_->CreateDeviceAddress(
    reinterpret_cast<void *>(ptr + offset_size), dst->Size(), dst->shape(), Format::DEFAULT_FORMAT, dst->data_type(),
    device_name, device_id, stream_id);

  MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(ptr) << ", size:" << dst->Size()
                << ", shape:" << dst->shape() << ", data_type:" << TypeIdToString(dst->data_type());
  MS_EXCEPTION_IF_NULL(dst_device_address);

  // set device address to dst
  dst->set_device_address(dst_device_address);
}

void RegReuseDataPtr(py::module *m) {
  (void)m->def("_reuse_data_ptr", &mindspore::ReuseDataPtr, "Reuse tensor device address.");
}
}  // namespace mindspore
