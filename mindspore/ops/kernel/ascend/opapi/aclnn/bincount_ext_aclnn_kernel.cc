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

#include "kernel/ascend/opapi/aclnn/bincount_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace bincount_ext {
namespace {
int64_t GetValueFromTensorToInt64(KernelTensor *tensor) {
  auto data_type = tensor->dtype_id();
  switch (data_type) {
    case kNumberTypeInt8:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int8_t>());
    case kNumberTypeInt16:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int16_t>());
    case kNumberTypeInt32:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int32_t>());
    case kNumberTypeInt64:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int64_t>());
    case kNumberTypeUInt8:
      return static_cast<int64_t>(tensor->GetValueWithCheck<uint8_t>());
    default:
      MS_LOG(EXCEPTION) << "Unsupported input data type: " << data_type;
  }
  return 0;
}

void SetKernelTensorShapeAndType(KernelTensor *tensor, const ShapeVector &shape_vector, const TypePtr &type_ptr) {
  tensor->SetType(type_ptr);
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape_vector));
}

void SetKernelTensorSize(KernelTensor *tensor) {
  size_t type_size = UnitSizeInBytes(tensor->dtype_id());
  size_t size = SizeOf(tensor->GetShapeVector()) * type_size;
  tensor->set_size(size);
}
}  // namespace

void BincountExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();

  input_dim_ = inputs[kIndex0]->GetShapeVector().size();
  input_numel_ = inputs[kIndex0]->GetShapeVector()[0];
  origin_output_typeptr_ = outputs[0]->GetType();

  min_length_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  outputs[kIndex0]->SetShapeVector({min_length_});
  GetWorkspaceForResizeBincount(inputs[kIndex0], inputs[kIndex1], min_length_, outputs[kIndex0]);
  // Check if null tensor
  if (!(input_dim_ == 1 && input_numel_ == 0)) {
    min_output_tensor_ = outputs[0];
    max_output_tensor_ = outputs[0];
    SetKernelTensorShapeAndType(min_output_tensor_, inputs[kIndex2]->GetShapeVector(), inputs[kIndex0]->GetType());
    SetKernelTensorShapeAndType(max_output_tensor_, inputs[kIndex2]->GetShapeVector(), inputs[kIndex0]->GetType());
    GetWorkspaceForResizeMin(inputs[kIndex0], min_output_tensor_);
    GetWorkspaceForResizeMax(inputs[kIndex0], max_output_tensor_);
  }
}

bool BincountExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run BincountExt start.";

  auto ouptut_shape = min_length_;

  // Check if null tensor
  if (!(input_dim_ == 1 && input_numel_ == 0)) {
    // Calculate min value to avoid negative integer
    RunOpMin(stream_ptr, workspace, inputs[kIndex0], min_output_tensor_);
    SetKernelTensorSize(min_output_tensor_);
    auto min_value = GetValueFromTensorToInt64(min_output_tensor_);
    if (min_value < 0) {
      MS_LOG(EXCEPTION) << "Bincount only supports non-negative input values.";
    }

    // Calculate max value for output shape
    RunOpMax(stream_ptr, workspace, inputs[kIndex0], max_output_tensor_);
    SetKernelTensorSize(max_output_tensor_);
    auto max_value = GetValueFromTensorToInt64(max_output_tensor_);

    ouptut_shape = max_value < min_length_ ? min_length_ : (max_value + 1);
    SetKernelTensorShapeAndType(outputs[kIndex0], {ouptut_shape}, origin_output_typeptr_);
    SetKernelTensorSize(outputs[kIndex0]);
  }
  // Update workspace and run aclnnBincount
  auto res = GEN_EXECUTOR_CUST(op_type_Bincount_, inputs[kIndex0], inputs[kIndex1], ouptut_shape, outputs[kIndex0]);
  UpdateWorkspace(res);
  executor_ = std::get<kIndex1>(res);
  auto op_Bincount_pair = GetExecutorBincount(inputs[kIndex0], inputs[kIndex1], ouptut_shape, outputs[kIndex0]);
  auto release_func = op_Bincount_pair.second;
  const auto &iter = ops_workspace_size_map_.find("Bincount");
  if (iter == ops_workspace_size_map_.end()) {
    RUN_OP_API_ASYNC(op_type_Bincount_, nullptr, 0, executor_, stream_ptr, release_func);
  } else {
    auto workspace_size_idx = iter->second.first;
    auto workspace_size = iter->second.second;
    if (workspace.empty() || workspace.size() <= workspace_size_idx) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    auto workspace_tensor = workspace[workspace_size_idx];
    if (workspace_tensor->size() != workspace_size) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API_ASYNC(op_type_Bincount_, workspace_tensor->device_ptr(), workspace_size, executor_, stream_ptr,
                     release_func);
  }

  // Update output shape
  output_shape_[kIndex0] = {ouptut_shape};

  MS_LOG(DEBUG) << "Run BincountExt end.";
  return true;
}

void BincountExtAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(output_shape_[kIndex0]);
}

MS_ACLNN_KERNEL_FACTORY_REG(BincountExt, BincountExtAscend);
}  // namespace bincount_ext
}  // namespace kernel
}  // namespace mindspore
