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

#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <utility>
#include <unordered_map>

#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "backend/common/optimizer/dynamic_shape_helper.h"
#include "kernel/ascend/opapi/aclnn_kernel_build.h"
#include "kernel/ascend/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_build.h"

#include "tensor/tensor.h"
#include "backend/ms_infer_backend/ms_kernel_lib.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {
namespace {
kernel::KernelModPtr CreateKernelMod(const PrimitivePtr &prim, const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  kernel::KernelModPtr kernel_mod_ptr;
  auto op_name = prim->name();

  // internal kernel
  kernel_mod_ptr = kernel::CreateInternalKernelMod(op_name, inputs, outputs);
  if (kernel_mod_ptr) {
    MS_LOG(INFO) << "Select internal kernel for op: " << op_name;
    return kernel_mod_ptr;
  }
  // aclnn kernel
  kernel_mod_ptr = kernel::CreateAclnnKernelMod(op_name);
  if (kernel_mod_ptr) {
    MS_LOG(INFO) << "Select aclnn kernel for op: " << op_name;
    return kernel_mod_ptr;
  }
  // rt kernel Reshape/ReshapeExt
  kernel_mod_ptr = kernel::CreateRtKernelMod(op_name);
  if (kernel_mod_ptr) {
    MS_LOG(INFO) << "Select rt kernel for op: " << op_name;
    return kernel_mod_ptr;
  }
  // acl kernel
  kernel_mod_ptr = kernel::CreateAclKernelMod(prim, inputs, outputs);
  if (kernel_mod_ptr) {
    MS_LOG(INFO) << "Select acl kernel for op: " << op_name;
    return kernel_mod_ptr;
  }

  return nullptr;
}

kernel::KernelModPtr SelectKernelMod(const PrimitivePtr &prim, const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  auto kernel_mod = CreateKernelMod(prim, inputs, outputs);
  if (kernel_mod == nullptr) {
    MS_LOG(EXCEPTION) << "Create kernelmod for op " << prim->name() << " failed";
  }

  if (!kernel_mod->Init(prim, inputs, outputs)) {
    MS_LOG(EXCEPTION) << "KernelMod Init failed: " << prim->name();
  }

  return kernel_mod;
}

void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                             const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(base_shape);
  size_t output_num = output_kernel_tensors.size();
  if (output_num > 1) {
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    const auto &shapes = sequence_shape->shape();
    if (shapes.size() != output_num) {
      MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << output_num
                        << ", but got: " << shapes.size();
    }
    for (size_t i = 0; i < output_num; i++) {
      const auto &kernel_tensor = output_kernel_tensors[i];
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->SetShapeVector(shapes[i]->GetShapeVector());
    }
  } else if (output_num == 1) {
    const auto &kernel_tensor = output_kernel_tensors[0];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    if ((kernel_tensor->type_id() != kObjectTypeTuple && kernel_tensor->type_id() != kObjectTypeList) &&
        sequence_shape != nullptr) {
      // For the operator prototype whose output is of type Tuple, the back-end operator is expanded as Tensors, and for
      // single-output scenarios, the InferShape result is TupleShape, and the back-end needs to expand it to
      // TensorShape. For example, the output of the split operator is only a Tensor scene.
      const auto &shapes = sequence_shape->shape();
      if (shapes.size() != 1) {
        MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << 1 << ", but got: " << shapes.size();
      }

      kernel_tensor->SetShapeVector(shapes[0]->GetShapeVector());
    } else {
      kernel_tensor->SetShapeVector(base_shape->GetShapeVector());
    }
  }
}
}  // namespace

DAKernelTensor::DAKernelTensor(da::tensor::DATensor *tensor) : tensor_(tensor) {
  MS_EXCEPTION_IF_NULL(tensor_);
  MS_LOG(INFO) << "New DAKernelTensor, DATensor: " << tensor << ", type: " << tensor->type;
  UpdateShapeVector(&shape_vector_, tensor_);

  // Set host_info_ for GetValue<> call in complex kernel mod, only set value for HOST_TENSOR
  if (tensor->tensorType == da::tensor::TensorType::HOST_TENSOR) {
    auto host_value = HostValueStore::GetInstance().GetValueByDATensor(tensor);
    auto host_value_abs = host_value->ToAbstract();
    MS_EXCEPTION_IF_NULL(host_value_abs);
    SetType(host_value_abs->GetType());
    SetShape(host_value_abs->GetShape());
    SetValue(host_value);
  } else {
    // currently only set object type for DEVICE_TENSOR/UNKNOW_TENSOR
    if (tensor->type == da::tensor::Type_Monad || tensor->type == da::tensor::Type_None) {
      SetType(TypeIdToType(ConvertDataType(tensor->type)));
    } else {
      SetType(TypeIdToType(kObjectTypeTensorType));
    }
    SetShape(std::make_shared<abstract::TensorShape>(shape_vector_));
  }
}

MsKernel::MsKernel(da::tensor::DATensor *tensor_node) : da::runtime::DAKernel(tensor_node) {
  MS_EXCEPTION_IF_NULL(tensorNode_);
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
     MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  device_context_->Initialize();
  MS_EXCEPTION_IF_NULL(device_context_);
  stream_ = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream_);
}

MsKernel::~MsKernel() {
  // Destroy kernel tensors
  auto destroy_tensors = [](std::vector<kernel::KernelTensor *> &kernel_tensors) {
    for (auto &tensor : kernel_tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      delete tensor;
    }
    kernel_tensors.clear();
  };
  destroy_tensors(outputs_);
  destroy_tensors(workspaces_);
}

void MsKernel::Init() {
  MS_EXCEPTION_IF_NULL(tensorNode_);

  CreateInputKernelTensors();
  CreateOutputKernelTensors();

  auto prim = HostValueStore::GetInstance().GetPrimByDATensor(tensorNode_);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Begin select kernelmod for Primitive: " << prim->name();
  kernel_mod_ = SelectKernelMod(prim, inputs_, outputs_);
  MS_EXCEPTION_IF_NULL(kernel_mod_);
}

void MsKernel::CreateInputKernelTensors() {
  MS_EXCEPTION_IF_NULL(tensorNode_);
  MS_LOG(INFO) << "Start create input DAKernelTensors";
  for (size_t i = 0; i < tensorNode_->inputSize; ++i) {
    MS_EXCEPTION_IF_NULL(tensorNode_->input[i]);
    if (tensorNode_->input[i]->type == da::tensor::Type_Monad) {
      continue;
    }
    auto input_tensor = std::make_shared<DAKernelTensor>(tensorNode_->input[i]);
    MS_EXCEPTION_IF_NULL(input_tensor);
    (void)abs_inputs_.emplace_back(input_tensor);
    (void)inputs_.emplace_back(input_tensor.get());
    MS_LOG(DEBUG) << "input kernel tensors: " << input_tensor->ToString();
  }
  MS_LOG(INFO) << "End create input DAKernelTensors";
}

void MsKernel::CreateOutputKernelTensors() {
  MS_EXCEPTION_IF_NULL(tensorNode_);
  MS_LOG(INFO) << "Start create output DAKernelTensors";
  if (tensorNode_->type == da::tensor::Type_Tensor) {
    auto **da_tensor_list = reinterpret_cast<da::tensor::DATensor **>(tensorNode_->data);
    MS_EXCEPTION_IF_NULL(da_tensor_list);
    for (size_t i = 0; i < tensorNode_->shape[0]; ++i) {
      auto output_tensor = new DAKernelTensor(da_tensor_list[i]);
      MS_EXCEPTION_IF_NULL(output_tensor);
      (void)outputs_.emplace_back(output_tensor);
      MS_LOG(DEBUG) << "Create output kernel tensor: " << output_tensor->ToString() << ", index: " << i;
    }
  } else {
    auto output_tensor = new DAKernelTensor(tensorNode_);
    MS_EXCEPTION_IF_NULL(output_tensor);
    (void)outputs_.emplace_back(output_tensor);
    MS_LOG(DEBUG) << "Create output kernel tensor: " << output_tensor->ToString();
  }
  MS_LOG(INFO) << "End create output DAKernelTensors";
}

void MsKernel::InferShape() {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // 1. Infer operator's output's Shape.
  MS_LOG(INFO) << "Begin InferShape for kernel: " << kernel_mod_->primitive() << ", inputs: " << abs_inputs_;
  auto base_shape = opt::dynamic_shape::InferShape(kernel_mod_->primitive(), abs_inputs_);
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_LOG(INFO) << "End InferShape for kernel: " << kernel_mod_->primitive() << ", shape: " << base_shape->ToString();

  // 2. Update shape of output kernel tensor.
  UpdateKernelTensorShape(base_shape, outputs_);
}

void MsKernel::Resize() {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  if (kernel_mod_->Resize(inputs_, outputs_) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "KernelMod Resize failed";
  }
}

void MsKernel::Launch() {
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  AllocateOutputDeviceMemory();
  AllocateWorkSpaceDeviceMemory();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  MS_EXCEPTION_IF_NULL(stream_);
  MS_LOG(INFO) << "Launch kernel " << kernel_mod_->kernel_name() << " start.";
  MS_LOG(INFO) << "inputs_: " << inputs_;
  MS_LOG(INFO) << "outputs_: " << outputs_;
  MS_LOG(INFO) << "workspaces_: " << workspaces_;
  MS_LOG(INFO) << "stream_: " << stream_;

  if (!kernel_mod_->Launch(inputs_, workspaces_, outputs_, stream_)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed.";
  }

  MS_LOG(INFO) << "Launch kernel completed.";
  FreeWorkSpaceDeviceMemory();
}

void MsKernel::AllocateOutputDeviceMemory() {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  MS_EXCEPTION_IF_NULL(device_context_);
  auto output_size_list = kernel_mod_->GetOutputSizeList();
  MS_EXCEPTION_IF_CHECK_FAIL(output_size_list.size() == outputs_.size(), "Output size is not same");
  for (size_t i = 0; i < output_size_list.size(); ++i) {
    auto output_data = device_context_->device_res_manager_->AllocateMemory(output_size_list[i], kDefaultStreamIndex);
    if (!output_data) {
      MS_LOG(EXCEPTION) << "Allocate output memory failed";
    }
    outputs_[i]->set_device_ptr(output_data);
  }
}

void MsKernel::AllocateWorkSpaceDeviceMemory() {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  MS_EXCEPTION_IF_NULL(device_context_);
  for (auto &size : kernel_mod_->GetWorkspaceSizeList()) {
    auto ktensor = new kernel::KernelTensor();
    MS_EXCEPTION_IF_NULL(ktensor);
    auto data = device_context_->device_res_manager_->AllocateMemory(size, kDefaultStreamIndex);
    if (!data) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    ktensor->set_size(size);
    ktensor->set_device_ptr(data);
    (void)workspaces_.emplace_back(ktensor);
  }
}

void MsKernel::FreeWorkSpaceDeviceMemory() {
  MS_EXCEPTION_IF_NULL(device_context_);
  for (auto &ws : workspaces_) {
    device_context_->device_res_manager_->FreeMemory(ws->device_ptr());
  }
  workspaces_.clear();
}

MsKernelLib::MsKernelLib() : da::runtime::KernelLib(std::move(kMindsporeKernelLibName)) {
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
     MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->Initialize();
}

da::runtime::DAKernel *MsKernelLib::CreateKernel(da::tensor::DATensor *tensor_node) const {
  MS_EXCEPTION_IF_NULL(tensor_node);
  return new (std::nothrow) MsKernel(tensor_node);
}

DART_REGISTER_KERNEL_LIB(kMindsporeKernelLibName, MsKernelLib);
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
