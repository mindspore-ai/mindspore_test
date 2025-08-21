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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/avg_pool1d_aclnn_kernel.h"

#include <tuple>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>

#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace avg_pool1d {
namespace {
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

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::tuple<bool, bool, int64_t, int8_t>>
AvgPool1DGenerate(const std::vector<KernelTensor *> &inputs) {
  auto kernel_size_val = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (kernel_size_val.size() > 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, kernel_size should contain one value but got "
                             << kernel_size_val.size();
  }

  auto stride_opt = inputs[kIndex2]->GetValue<std::vector<int64_t>>();
  auto stride_val = stride_opt.has_value() ? stride_opt.value() : kernel_size_val;
  if (stride_val.size() > 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, stride should contain one value but got " << stride_val.size();
  }

  auto padding_val = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  if (padding_val.size() > 1) {
    MS_EXCEPTION(ValueError) << "For Op AvgPool1d, padding should contain one value but got " << padding_val.size();
  }

  bool ceil_mode = inputs[kIndex4]->GetValueWithCheck<bool>();
  bool count_include_pad = inputs[kIndex5]->GetValueWithCheck<bool>();
  int64_t divisor_override = 0;
  int8_t cube_math_type = OpApiUtil::GetCubeMathType();
  std::tuple<bool, bool, int64_t, int8_t> param =
    std::make_tuple(ceil_mode, count_include_pad, divisor_override, cube_math_type);
  return std::make_tuple(std::vector<int64_t>{1, kernel_size_val[0]}, std::vector<int64_t>{1, stride_val[0]},
                         std::vector<int64_t>{0, padding_val[0]}, param);
}
}  // namespace

void AvgPool1DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto expand_shape = in_shape;
  expand_shape.insert(expand_shape.end() - 1, 1);
  input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
  SetTensorStorageInfo<std::shared_ptr<KernelTensor>>(input_kernel_tensor_, expand_shape);

  auto params = AvgPool1DGenerate(inputs);
  const auto &kernel_size = std::get<0>(params);
  const auto &stride = std::get<1>(params);
  const auto &padding = std::get<2>(params);
  auto [ceil_mode, count_include_pad, divisor_override, cube_math_type] = std::get<3>(params);

  auto out_shape = outputs[kIndex0]->GetShapeVector();
  output_kernel_tensor_ = outputs[kIndex0]->CloneKernelTensor();
  ShapeVector expand_out_shape = out_shape;
  expand_out_shape.insert(expand_out_shape.end() - 1, 1);
  SetTensorStorageInfo<std::shared_ptr<KernelTensor>>(output_kernel_tensor_, expand_out_shape);
  GetWorkspaceForResize(input_kernel_tensor_.get(), kernel_size, stride, padding, ceil_mode, count_include_pad,
                        divisor_override, cube_math_type, output_kernel_tensor_.get());
}

bool AvgPool1DAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto params = AvgPool1DGenerate(inputs);
  const auto &kernel_size = std::get<0>(params);
  const auto &stride = std::get<1>(params);
  const auto &padding = std::get<2>(params);
  auto [ceil_mode, count_include_pad, divisor_override, cube_math_type] = std::get<3>(params);
  input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
  output_kernel_tensor_->set_device_ptr(outputs[kIndex0]->device_ptr());
  RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), kernel_size, stride, padding, ceil_mode, count_include_pad,
        divisor_override, cube_math_type, output_kernel_tensor_.get());

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AvgPool1D, AvgPool1DAscend);
}  // namespace avg_pool1d
}  // namespace kernel
}  // namespace mindspore
