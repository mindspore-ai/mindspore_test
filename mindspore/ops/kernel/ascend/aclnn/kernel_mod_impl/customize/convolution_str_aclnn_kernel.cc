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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/convolution_str_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace convolution_str {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim) {
  if (param->size() == kIndex1) {
    param->insert(param->end(), expect_dim - kIndex1, param->at(kIndex0));
  }
}
}  // namespace
void ConvolutionStrAscend::SetExpandTensor(KernelTensor *input_tensor, const std::vector<KernelTensor *> &inputs) {
  input_tensor->SetType(inputs[kIndex0]->GetType());
  input_tensor->SetShape(std::make_shared<abstract::TensorShape>(pad_nd_shape_));
}

void ConvolutionStrAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  expand_indices_.clear();
  auto spatial_len = inputs[kIndex1]->GetShapeVector().size() - kIndex2;

  stride_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  ExpandParamIfNeeded(&stride_, spatial_len);
  padding_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  dilation_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  ExpandParamIfNeeded(&dilation_, spatial_len);
  transposed_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex6]);
  output_padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex7]);
  groups_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex8]);

  auto &input_sizes = inputs[kIndex0]->GetShape()->GetShapeVector();
  auto &weight_sizes = inputs[kIndex1]->GetShape()->GetShapeVector();
  auto k = SizeToLong(weight_sizes.size());
  auto dim = static_cast<size_t>(k - 2);
  pad_vector_ = std::vector<int64_t>(dim, 0);
  if (padding_ == PadMode::SAME) {
    std::vector<int64_t> padding_l;
    std::vector<int64_t> padding_r;
    bool symmetric_padding = true;
    for (size_t i = 0; i < dim; ++i) {
      auto stride = stride_.size() == 1 ? stride_[0] : stride_[i];
      auto dilation = dilation_.size() == 1 ? dilation_[0] : dilation_[i];
      auto inputSize = input_sizes[i + 2];
      auto kernelSize = weight_sizes[i + 2];
      auto total_padding = dilation * (kernelSize - 1);
      if (stride > 2 && (total_padding % 2 == 1)) {
        auto wiggle_room = inputSize % stride - 1;
        if (wiggle_room > 0) {
          --total_padding;
        }
      }
      auto left = total_padding / 2;
      auto right = total_padding - left;
      padding_l.push_back(left);
      padding_r.push_back(right);
      if (left != right) {
        symmetric_padding = false;
      }
    }
    if (symmetric_padding) {
      pad_vector_ = padding_l;
      GetWorkspaceForResizeConvolutionStr(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, pad_vector_,
                                          dilation_, transposed_, output_padding_, groups_, outputs[kIndex0],
                                          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
    } else {
      pad_nd_.resize(2 * dim, 0);
      for (size_t i = 0; i < dim; ++i) {
        auto delta_pad = padding_r[i] - padding_l[i];
        auto pad_idx = 2 * (dim - 1 - i);
        if (delta_pad > 0) {
          pad_nd_[pad_idx + 1] = delta_pad;
        } else {
          pad_nd_[pad_idx] = delta_pad;
          padding_l[i] = padding_r[i];
        }
      }

      pad_vector_ = padding_l;
      need_ConstantPadNd_ = true;

      // infer outshape
      pad_nd_shape_ = std::vector<int64_t>{};
      std::vector<int64_t> x_shape = inputs[kIndex0]->GetShapeVector();
      auto x_rank = x_shape.size();
      size_t kScaleNum = 2;
      auto l_diff = x_rank - (pad_nd_.size() / kScaleNum);
      for (size_t i = 0; i < l_diff; ++i) {
        (void)pad_nd_shape_.emplace_back(x_shape[i]);
      }
      for (size_t i = 0; i < pad_nd_.size() / kScaleNum; ++i) {
        auto pad_idx = pad_nd_.size() - ((i + 1) * 2);
        auto new_dim = x_shape[l_diff + i] + pad_nd_[pad_idx] + pad_nd_[pad_idx + 1];
        (void)pad_nd_shape_.emplace_back(new_dim);
      }

      SetExpandTensor(&input_expand_, inputs);
      KernelTensor *input_ptr = &input_expand_;
      GetWorkspaceForResizeConstantPadNd(inputs[kIndex0], pad_nd_, zero_, &input_expand_);
      expand_indices_.emplace_back(kIndex0);
      GetWorkspaceForResizeConvolutionStr(input_ptr, inputs[kIndex1], inputs[kIndex2], stride_, pad_vector_, dilation_,
                                          transposed_, output_padding_, groups_, outputs[kIndex0],
                                          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));

      expand_count_ = expand_indices_.size();
      for (size_t i = 0; i < expand_count_; i++) {
        auto type_size = mindspore::abstract::TypeIdSize(inputs[expand_indices_[i]]->dtype_id());
        size_t output_size = 1;
        for (const int64_t &size_value : pad_nd_shape_) {
          // Casting each int64_t value to size_t during multiplication
          output_size *= static_cast<size_t>(size_value);
        }
        output_size *= type_size;

        workspace_size_list_.emplace_back(output_size);
      }
    }
  } else if (padding_ == PadMode::VALID) {
    GetWorkspaceForResizeConvolutionStr(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, pad_vector_,
                                        dilation_, transposed_, output_padding_, groups_, outputs[kIndex0],
                                        OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  } else {
    MS_LOG(EXCEPTION) << "Input padding string must be one of {'same', 'valid'}";
  }
}

bool ConvolutionStrAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (need_ConstantPadNd_) {
    KernelTensor *output_ptr;
    input_expand_.set_device_ptr(workspace[workspace.size() - expand_count_]->device_ptr());
    output_ptr = &input_expand_;
    KernelTensor *input_ptr = &input_expand_;
    RunOpConstantPadNd(stream_ptr, workspace, inputs[kIndex0], pad_nd_, zero_, output_ptr);

    RunOpConvolutionStr(stream_ptr, workspace, input_ptr, inputs[kIndex1], inputs[kIndex2], stride_, pad_vector_,
                        dilation_, transposed_, output_padding_, groups_, outputs[kIndex0],
                        OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  } else {
    RunOpConvolutionStr(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, pad_vector_,
                        dilation_, transposed_, output_padding_, groups_, outputs[kIndex0],
                        OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ConvolutionStr, ConvolutionStrAscend);
}  // namespace convolution_str
}  // namespace kernel
}  // namespace mindspore
