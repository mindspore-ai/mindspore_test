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

#include "common/kernel_tensor.h"
#include "common/kernel_mod_cache.h"
#include "common/ms_factory.h"
#include "common/format_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "kernel/ascend/opapi/aclnn_kernel_build.h"
#include "kernel/ascend/acl/acl_kernel_build.h"

#include "dalang/dair/tensor/tensor.h"
#include "backend/ms_infer_backend/ms_kernel_lib.h"
#include "backend/ms_infer_backend/host_value_store.h"
#include "backend/ms_infer_backend/utils.h"

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
  if (kernel_mod->Resize(inputs, outputs) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "KernelMod Resize failed";
  }

  return kernel_mod;
}
}  // namespace

// DAKernelTensor is a KernelTensor that wraps a DATensor.
class DAKernelTensor : public kernel::KernelTensor {
 public:
  DAKernelTensor() = delete;
  ~DAKernelTensor() = default;

  explicit DAKernelTensor(da::tensor::DATensor *tensor) : tensor_(tensor) {
    MS_EXCEPTION_IF_NULL(tensor_);
    MS_LOG(INFO) << "New DAKernelTensor, DATensor: " << tensor;
    for (size_t i = 0; i < tensor_->dim; i++) {
      (void)shape_vector_.emplace_back(tensor_->shape[i]);
    }

    // Set host_info_ for GetValue<> call in complex kernel mod, only set value for HOST_TENSOR
    if (tensor->tensorType == da::tensor::TensorType::HOST_TENSOR) {
      auto host_value = HostValueStore::GetInstance().Get(tensor);
      auto host_value_abs = host_value->ToAbstract();
      MS_EXCEPTION_IF_NULL(host_value_abs);
      SetType(host_value_abs->GetType());
      SetShape(host_value_abs->GetShape());
      SetValue(host_value);
    } else {
      // currently only set object type for DEVICE_TENSOR/UNKNOW_TENSOR
      SetType(TypeIdToType(kObjectTypeTensorType));
      SetShape(std::make_shared<abstract::TensorShape>(shape_vector_));
    }
  }

  // Set the shape vector for Tensor/Sequence/Scalar.
  void SetShapeVector(const ShapeVector &shape_vector) override {
    SetTensorShape(tensor_, shape_vector);
    shape_vector_ = shape_vector;
    SetShape(std::make_shared<abstract::TensorShape>(shape_vector_));
  }

  // Set the shape vector for Tensor/Sequence/Scalar with rvalue.
  void SetShapeVector(ShapeVector &&shape_vector) override {
    SetTensorShape(tensor_, shape_vector);
    shape_vector_ = std::move(shape_vector);
    SetShape(std::make_shared<abstract::TensorShape>(shape_vector_));
  }

  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const override { return shape_vector_; }

  // Get the device shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetDeviceShapeVector() const override { return shape_vector_; }

  // Get the data enum type id of the KernelTensor.
  TypeId dtype_id() const override { return ConvertDataType(tensor_->type); }

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const override { return tensor_->data; }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr) override { tensor_->data = ptr; }

  // Get the memory size in byte of the KernelTensor.
  size_t size() const override {
    return da::tensor::DataTypeSize(tensor_->type) * da::tensor::ShapeSize(tensor_->shape);
  }

  // Get string representation of tensor format
  std::string GetStringFormat() const override { return kernel::GetFormatFromEnumToStr(format_); }
  void set_format(mindspore::Format format) override { format_ = format; }

 private:
  da::tensor::DATensor *tensor_{nullptr};
  ShapeVector shape_vector_;
  mindspore::Format format_{mindspore::Format::DEFAULT_FORMAT};
};

class DAKernel {
 public:
  explicit DAKernel(da::tensor::DATensor *da_tensor, device::DeviceContext *device_context);
  ~DAKernel();

  void AllocateOutputDeviceMemory();
  void AllocateWorkspaceDeviceMemory();
  void FreeWorkspaceDeviceMemory();
  void Launch();

 private:
  kernel::KernelModPtr kernel_mod_;
  std::vector<kernel::KernelTensor *> inputs_;
  std::vector<kernel::KernelTensor *> outputs_;
  std::vector<kernel::KernelTensor *> workspaces_;
  void *stream_{nullptr};
  device::DeviceContext *device_context_{nullptr};
};

DAKernel::DAKernel(da::tensor::DATensor *da_tensor, device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(da_tensor);
  MS_EXCEPTION_IF_NULL(device_context);
  device_context_ = device_context;
  stream_ = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream_);

  auto prim = ConvertPrimitiveOp(da_tensor->op);
  MS_LOG(DEBUG) << "Primitive: " << prim->name();

  // Initialize input kernel tensors
  for (size_t i = 0; i < da_tensor->inputSize; ++i) {
    auto input_tensor = new DAKernelTensor(da_tensor->input[i]);
    MS_EXCEPTION_IF_NULL(input_tensor);
    (void)inputs_.emplace_back(input_tensor);
    MS_LOG(DEBUG) << "input kernel tensors: " << input_tensor->ToString();
  }
  // Initialize output kernel tensor
  auto output_tensor = new DAKernelTensor(da_tensor);
  MS_EXCEPTION_IF_NULL(output_tensor);
  (void)outputs_.emplace_back(output_tensor);
  MS_LOG(DEBUG) << "output kernel tensor: " << output_tensor->ToString();

  // Create KernelMod
  kernel_mod_ = SelectKernelMod(prim, inputs_, outputs_);
}

DAKernel::~DAKernel() {
  // Destroy kernel tensors
  auto destroy_tensors = [](std::vector<kernel::KernelTensor *> &kernel_tensors) {
    for (auto &tensor : kernel_tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      delete tensor;
    }
    kernel_tensors.clear();
  };
  destroy_tensors(inputs_);
  destroy_tensors(outputs_);
  destroy_tensors(workspaces_);
}

void DAKernel::AllocateOutputDeviceMemory() {
  MS_EXCEPTION_IF_CHECK_FAIL(kernel_mod_->GetOutputSizeList().size() == 1, "Invalid kernel mod output size");
  auto output_size = kernel_mod_->GetOutputSizeList()[0];
  auto output_data = device_context_->device_res_manager_->AllocateMemory(output_size, kDefaultStreamIndex);
  if (!output_data) {
    MS_LOG(EXCEPTION) << "Allocate output memory failed";
  }
  outputs_[0]->set_device_ptr(output_data);
}

void DAKernel::AllocateWorkspaceDeviceMemory() {
  // Allocate workspace device memory
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

void DAKernel::FreeWorkspaceDeviceMemory() {
  for (auto &ws : workspaces_) {
    device_context_->device_res_manager_->FreeMemory(ws->device_ptr());
  }
  workspaces_.clear();
}

void DAKernel::Launch() {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  MS_LOG(INFO) << "Launch kernel " << kernel_mod_->kernel_name() << " start.";
  MS_LOG(INFO) << "inputs_: " << inputs_;
  MS_LOG(INFO) << "outputs_: " << outputs_;
  MS_LOG(INFO) << "workspaces_: " << workspaces_;
  MS_LOG(INFO) << "stream_: " << stream_;

  if (!kernel_mod_->Launch(inputs_, workspaces_, outputs_, stream_)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed.";
  }
  MS_LOG(INFO) << "Launch kernel completed.";
}

bool MindsporeKernelLib::RunTensor(da::tensor::DATensor *tensor, da::runtime::MemoryPool *mempool) const {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context_);

  MS_LOG(INFO) << "Run tensor start.";

  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);

  static std::unordered_map<da::tensor::DATensor *, std::shared_ptr<DAKernel>> kernel_cache;
  if (kernel_cache.find(tensor) == kernel_cache.end()) {
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);

    auto kernel = std::make_shared<DAKernel>(tensor, device_context_);
    kernel->AllocateOutputDeviceMemory();
    kernel_cache[tensor] = kernel;
  }

  kernel_cache[tensor]->AllocateWorkspaceDeviceMemory();
  kernel_cache[tensor]->Launch();
  kernel_cache[tensor]->FreeWorkspaceDeviceMemory();

  MS_LOG(INFO) << "Run tensor end.";

  return true;
}

DART_REGISTER_KERNEL_LIB(kMindsporeKernelLibName, MindsporeKernelLib);
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
