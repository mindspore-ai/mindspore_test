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
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

#include "dalang/dair/tensor/tensor.h"
#include "backend/ms_infer_backend/ms_kernel_lib.h"
#include "backend/ms_infer_backend/utils.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

// DAKernelTensor is a KernelTensor that wraps a DATensor.
class DAKernelTensor : public kernel::KernelTensor {
 public:
  DAKernelTensor() = delete;
  ~DAKernelTensor() = default;

  explicit DAKernelTensor(da::tensor::DATensor *tensor) : tensor_(tensor) {
    MS_EXCEPTION_IF_NULL(tensor_);
    for (size_t i = 0; i < tensor_->dim; i++) {
      (void)shape_vector_.emplace_back(tensor_->shape[i]);
    }

    // Set host_info_ for GetValue<> call in complex kernel mod.
    auto dtype = ConvertDataType(tensor_->type);
    SetType(TypeIdToType(dtype));
    SetShape(std::make_shared<abstract::TensorShape>(shape_vector_));
    if (shape_vector_.size() <= 1) {  // only copy data for small tensor
      auto value = std::make_shared<tensor::Tensor>(dtype, shape_vector_, tensor_->data, dtype);
      SetValue(value);
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

  // Get the object enum type id of the KernelTensor.
  TypeId type_id() const override { return ConvertDataType(tensor_->type); }

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

 private:
  da::tensor::DATensor *tensor_{nullptr};
  ShapeVector shape_vector_;
};

static kernel::KernelModPtr CreateKernelMod(PrimitivePtr prim, std::vector<kernel::KernelTensor *> inputs,
                                            std::vector<kernel::KernelTensor *> outputs) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
     MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});

  const auto &op_name = prim->name();
  auto kernel_mod_ = device_context->GetKernelExecutor(false)->CreateKernelMod(op_name);
  if (kernel_mod_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create kernelmod for op " << op_name << " failed";
  }

  if (!kernel_mod_->Init(prim, inputs, outputs)) {
    MS_LOG(EXCEPTION) << "KernelMod Init failed: " << op_name;
  }
  if (kernel_mod_->Resize(inputs, outputs) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "KernelMod Resize failed";
  }

  return kernel_mod_;
}

class DAKernel {
 public:
  explicit DAKernel(da::tensor::DATensor *da_tensor);
  ~DAKernel();

  void AllocateDeviceMemory(da::runtime::MemoryPool *mempool);
  void Launch();

 private:
  kernel::KernelModPtr kernel_mod_;
  std::vector<kernel::KernelTensor *> inputs_;
  std::vector<kernel::KernelTensor *> outputs_;
  std::vector<kernel::KernelTensor *> workspaces_;
};

DAKernel::DAKernel(da::tensor::DATensor *da_tensor) {
  MS_EXCEPTION_IF_NULL(da_tensor);

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
  kernel_mod_ = CreateKernelMod(prim, inputs_, outputs_);
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

void DAKernel::AllocateDeviceMemory(da::runtime::MemoryPool *mempool) {
  MS_EXCEPTION_IF_CHECK_FAIL(kernel_mod_->GetOutputSizeList().size() == 1, "Invalid kernel mod output size");
  auto output_size = kernel_mod_->GetOutputSizeList()[0];
  auto output_data = mempool->Allocate(output_size);
  if (!output_data) {
    MS_LOG(EXCEPTION) << "Allocate output memory failed";
  }
  outputs_[0]->set_device_ptr(output_data);

  // Allocate workspace device memory
  for (auto &size : kernel_mod_->GetWorkspaceSizeList()) {
    auto ktensor = new kernel::KernelTensor();
    MS_EXCEPTION_IF_NULL(ktensor);
    auto data = mempool->Allocate(size);
    if (!data) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    ktensor->set_device_ptr(data);
    (void)workspaces_.emplace_back(ktensor);
  }
}

void DAKernel::Launch() {
  MS_LOG(DEBUG) << "Launch kernel start.";
  if (!kernel_mod_->Launch(inputs_, workspaces_, outputs_, nullptr)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed.";
  }
  MS_LOG(DEBUG) << "Launch kernel completed.";
}

bool MindsporeKernelLib::RunTensor(da::tensor::DATensor *tensor, da::runtime::MemoryPool *mempool) const {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(mempool);

  static std::unordered_map<da::tensor::DATensor *, std::shared_ptr<DAKernel>> kernel_cache;
  if (kernel_cache.find(tensor) == kernel_cache.end()) {
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);

    auto kernel = std::make_shared<DAKernel>(tensor);
    kernel->AllocateDeviceMemory(mempool);
    kernel_cache[tensor] = kernel;
  }

  kernel_cache[tensor]->Launch();

  return true;
}

DART_REGISTER_KERNEL_LIB(kMindsporeKernelLibName, MindsporeKernelLib);
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
