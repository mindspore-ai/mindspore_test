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
#include "kernel/ascend/opapi/aclnn/adaptive_avg_pool1d_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

TensorStorageInfoPtr CreateTensorStorageInfoPtr(const std::vector<int64_t> &shape) {
  size_t offset = 0;
  const std::vector<int64_t> expand_shape_ori = shape;
  const std::vector<int64_t> expand_shape_new = shape;
  auto expand_stride_ori = GetOriStrides(expand_shape_ori);
  auto expand_stride_new = expand_stride_ori;
  return std::make_shared<TensorStorageInfo>(expand_shape_new, expand_stride_new, offset, expand_shape_ori,
                                             expand_stride_ori, true);
}

template <typename T>
void SetTensorStorageInfo(T kernel_tensor, ShapeVector shape) {
  kernel_tensor->SetShapeVector(shape);
  TensorStorageInfoPtr tensor_storage_info = CreateTensorStorageInfoPtr(shape);
  kernel_tensor->set_tensor_storage_info(tensor_storage_info);
}

void AdaptiveAvgPool1DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto output_size_value_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  auto output_size = output_size_value_opt.value();
  auto expand_shape = in_shape;
  expand_shape.insert(expand_shape.end() - 1, 1);
  input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
  SetTensorStorageInfo<std::shared_ptr<KernelTensor>>(input_kernel_tensor_, expand_shape);

  auto out_shape = outputs[kIndex0]->GetShapeVector();
  auto out_shape_ori = out_shape;
  ShapeVector expand_out_shape = out_shape;
  expand_out_shape.insert(expand_out_shape.end() - 1, 1);
  SetTensorStorageInfo<KernelTensor *>(outputs[kIndex0], expand_out_shape);

  output_size_for_2d_ = {1, output_size};
  GetWorkspaceForResize(input_kernel_tensor_.get(), output_size_for_2d_, outputs[kIndex0]);
  SetTensorStorageInfo<KernelTensor *>(outputs[kIndex0], out_shape_ori);
}

bool AdaptiveAvgPool1DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
  RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), output_size_for_2d_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveAvgPool1D, AdaptiveAvgPool1DAscend);
}  // namespace kernel
}  // namespace mindspore
