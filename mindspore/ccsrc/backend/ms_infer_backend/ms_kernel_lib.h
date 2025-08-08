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

#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H_

#include <string>
#include <vector>
#include <utility>
#include <memory>

#include "ir/anf.h"
#include "ir/dtype.h"
#include "common/kernel.h"
#include "common/kernel_tensor.h"
#include "common/format_utils.h"
#include "runtime/hardware/device_context.h"

#include "tensor/tensor.h"
#include "runtime/kernel.h"
#include "runtime/kernel_lib.h"
#include "backend/ms_infer_backend/utils.h"
#include "backend/ms_infer_backend/host_value_store.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

const char kMindsporeKernelLibName[] = "Mindspore";

class MsKernel : public da::runtime::DAKernel {
 public:
  explicit MsKernel(da::tensor::DATensor *tensor_node);
  ~MsKernel() override = default;

  void Init() override;
  void InferShape() override;
  void Resize() override;
  void Launch() override;

 private:
  void AllocateOutputDeviceMemory();
  void AllocateWorkSpaceDeviceMemory();
  void FreeWorkSpaceDeviceMemory();
  void CreateInputKernelTensors();
  void CreateOutputKernelTensors();

  kernel::KernelModPtr kernel_mod_;
  AbstractBasePtrList abs_inputs_;
  std::vector<kernel::KernelTensor *> inputs_;
  std::vector<kernel::KernelTensor *> outputs_;
  std::vector<kernel::KernelTensor *> workspaces_;
  std::vector<kernel::KernelTensorPtr> input_kernel_tensors_;
  std::vector<kernel::KernelTensorPtr> output_kernel_tensors_;
  std::vector<kernel::KernelTensorPtr> workspace_kernel_tensors_;
  void *stream_{nullptr};
  device::DeviceContext *device_context_{nullptr};
};

class MsKernelLib : public da::runtime::KernelLib {
 public:
  MsKernelLib();
  ~MsKernelLib() override = default;

  da::runtime::DAKernel *CreateKernel(da::tensor::DATensor *tensor_node) const override;

 private:
  device::DeviceContext *device_context_{nullptr};
};

// DAKernelTensor is a KernelTensor that wraps a DATensor.
class DAKernelTensor : public kernel::KernelTensor {
 public:
  DAKernelTensor() = delete;
  ~DAKernelTensor() override = default;

  explicit DAKernelTensor(da::tensor::DATensor *tensor);

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

  // Get the BaseShape for Tensor/Sequence/Scalar.
  BaseShapePtr GetShape() const override {
    auto base_shape = KernelTensor::GetShape();
    MS_EXCEPTION_IF_NULL(base_shape);
    MS_EXCEPTION_IF_NULL(tensor_);
    if (!base_shape->isa<TensorShape>()) {
      return base_shape;
    }
    ShapeVector shape_vector;
    UpdateShapeVector(&shape_vector, tensor_);
    base_shape->SetShapeVector(shape_vector);
    return base_shape;
  }
  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const override {
    auto base_shape = GetShape();
    MS_EXCEPTION_IF_NULL(base_shape);
    return base_shape->GetShapeVector();
  }

  // Get the device shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetDeviceShapeVector() const override { return GetShapeVector(); }

  // Get the data enum type id of the KernelTensor.
  TypeId dtype_id() const override { return ConvertDataType(tensor_->type); }

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const override { return tensor_->data; }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr) override { tensor_->data = ptr; }

  // Get the memory size in byte of the KernelTensor.
  size_t size() const override {
    return UnitSizeInBytes(ConvertDataType(tensor_->type)) * da::tensor::ShapeSize(tensor_->shape);
  }

  // Get string representation of tensor format
  std::string GetStringFormat() const override { return kernel::GetFormatFromEnumToStr(format_); }
  void set_format(mindspore::Format format) override { format_ = format; }

 private:
  da::tensor::DATensor *tensor_{nullptr};
  ShapeVector shape_vector_;
  mindspore::Format format_{mindspore::Format::DEFAULT_FORMAT};
};
}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H_
