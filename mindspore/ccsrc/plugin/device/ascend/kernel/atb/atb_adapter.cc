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
#include "plugin/device/ascend/kernel/atb/atb_adapter.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore::device::ascend {
namespace {
atb::Tensor GetAtbTensor(mindspore::kernel::KernelTensor *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  atb::Tensor tensor;
  if (kernel_tensor == nullptr) {
    return tensor;
  }
  const auto &shape = kernel_tensor->GetShapeVector();
  const auto shape_size = shape.size();
  tensor.desc.dtype = AclConverter::ConvertType(kernel_tensor->dtype_id());
  tensor.desc.format = ACL_FORMAT_ND;
  tensor.desc.shape.dimNum = shape_size;
  for (size_t i = 0; i < shape_size; i++) {
    tensor.desc.shape.dims[i] = shape[i];
  }
  tensor.dataSize = kernel_tensor->size();
  // only contiguous tensor is supported now.
  tensor.deviceData = kernel_tensor->device_ptr();
  return tensor;
}

void UpdateAddress(mindspore::kernel::KernelTensor *kernel_tensor, atb::Tensor *tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->deviceData = kernel_tensor->device_ptr();
}
}  // namespace

static atb::Context *atb_context = nullptr;

ParamSetter &ParamSetter::Input(mindspore::kernel::KernelTensor *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  atb::Tensor tensor = GetAtbTensor(kernel_tensor);
  variant_pack.inTensors.push_back(tensor);
  return *this;
}

ParamSetter &ParamSetter::Input(std::optional<mindspore::kernel::KernelTensor *> kernel_tensor) {
  if (kernel_tensor.has_value()) {
    return Input(kernel_tensor.value());
  }
  return Input(nullptr);
}

ParamSetter &ParamSetter::Output(mindspore::kernel::KernelTensor *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  atb::Tensor tensor = GetAtbTensor(kernel_tensor);
  variant_pack.outTensors.push_back(tensor);

  if (stream == nullptr) {
    size_t stream_id = kernel_tensor->stream_id();
    stream = device::ascend::AscendStreamMng::GetInstance().GetStream(stream_id);
  }
  return *this;
}

ParamSetter &ParamSetter::Output(std::optional<mindspore::kernel::KernelTensor *> kernel_tensor) {
  if (kernel_tensor.has_value()) {
    return Output(kernel_tensor.value());
  }
  return Output(nullptr);
}

void ParamSetter::Update(const std::vector<mindspore::kernel::KernelTensor *> &inputs,
                         const std::vector<mindspore::kernel::KernelTensor *> &outputs) {
  for (size_t i = 0; i < input_ids.size(); ++i) {
    UpdateAddress(inputs[input_ids[i]], &(variant_pack.inTensors[i]));
  }
  for (size_t i = 0; i < output_ids.size(); ++i) {
    UpdateAddress(outputs[output_ids[i]], &(variant_pack.outTensors[i]));
  }
}

uint64_t GetWorkSpaceSize(atb::Operation *op, atb::VariantPack variant_pack, aclrtStream stream) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(stream);
  uint64_t workspace_size;
  auto ret = op->Setup(variant_pack, workspace_size, GetAtbContext(stream));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Setup failed.";
  }
  return workspace_size;
}

void Launch(atb::Operation *op, atb::VariantPack variant_pack, void *workspace_ptr,
            std::vector<size_t> workpsace_size_list, aclrtStream stream) {
  MS_EXCEPTION_IF_NULL(op);
  auto ret = op->Execute(variant_pack, reinterpret_cast<uint8_t *>(workspace_ptr), workpsace_size_list[0],
                         GetAtbContext(stream));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Execute failed.";
  }
}

atb::Context *GetAtbContext(const aclrtStream &stream) {
  if (atb_context == nullptr) {
    auto create_status = atb::CreateContext(&atb_context);
    if (create_status != 0) {
      MS_LOG(EXCEPTION) << "Create atb context failed.";
    }
    auto set_status = atb_context->SetExecuteStream(stream);
    if (set_status != 0) {
      MS_LOG(EXCEPTION) << "Set atb context stream failed.";
    }
  }
  return atb_context;
}
}  // namespace mindspore::device::ascend
