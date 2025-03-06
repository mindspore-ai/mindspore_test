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
#include "kernel/ascend/opapi/aclnn/dense_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace dense {
void DenseAclnnKernelMod::SetFlatternNdLinearTensorStorageInfo(const KernelTensorPtr &new_tensor,
                                                               const int &new_shape_first, const ShapeVector &shape) {
  auto new_shape_second = shape[shape.size() - 1];
  auto new_shape = ShapeVector{new_shape_first, new_shape_second};

  new_tensor->SetShapeVector(new_shape);

  size_t offset = 0;
  auto shape_ori = new_shape;
  auto shape_new = new_shape;
  const std::vector<int64_t> strides_new = {new_shape_second, 1};
  const std::vector<int64_t> strides_ori = strides_new;
  TensorStorageInfoPtr tensor_storage_info =
    std::make_shared<TensorStorageInfo>(shape_new, strides_new, offset, shape_ori, strides_ori, true);

  new_tensor->set_tensor_storage_info(tensor_storage_info);
}

void DenseAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  auto w_shape = inputs[kIndex1]->GetShapeVector();

  auto w_rank = w_shape.size();
  auto x_rank = x_shape.size();
  // 1. Generate Transpose weight, the weight'rank greater than 0
  ShapeVector w_t_shape = w_shape;
  if (w_rank < kDim2) {
    t_perm_.resize(kDim1);
    t_perm_[0] = 0;
  } else {
    t_perm_.resize(w_rank);
    // w_t modifies the last 2 dimensions, leaving the rest intact.
    t_perm_[w_rank - kDim1] = w_rank - kDim2;
    t_perm_[w_rank - kDim2] = w_rank - kDim1;
    w_t_shape[w_rank - kDim1] = w_shape[w_rank - kDim2];
    w_t_shape[w_rank - kDim2] = w_shape[w_rank - kDim1];

    for (size_t i = 0; i < w_rank - kDim2; ++i) {
      t_perm_[i] = i;
    }
  }

  auto w_t_ = &w_t_tensor_;
  w_t_tensor_.SetType(inputs[kIndex1]->GetType());
  w_t_tensor_.SetShape(std::make_shared<abstract::TensorShape>(w_t_shape));
  GetWorkspaceForResizeTransform(inputs[kIndex1], t_perm_, &w_t_tensor_);
  // 2. Go through Different scenarios based on input'rank and bias'rank.
  auto bias = inputs[kIndex2];
  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowMatmulHF32());

  MAKE_SCALAR(1, kNumberTypeInt64, one_);
  if (bias->GetType()->type_id() == kMetaTypeNone) {
    GetWorkspaceForResizeMatmul(inputs[kIndex0], w_t_, outputs[kIndex0], cube_math_type_);
  } else {
    auto bias_shape = bias->GetShapeVector();
    auto bias_rank = bias_shape.size();

    if (x_rank == kDim2) {
      GetWorkspaceForResizeAddmm(inputs[kIndex2], inputs[kIndex0], w_t_, one_, one_, outputs[kIndex0], cube_math_type_);
    } else if (bias_rank == kDim1 || x_rank == kDim3) {
      input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
      output_kernel_tensor_ = outputs[kIndex0]->CloneKernelTensor();

      int input_reshape_size = 1;
      if (x_rank > kDim1) {
        input_reshape_size = std::accumulate(x_shape.begin(), x_shape.end() - 1, 1, std::multiplies<int64_t>());
      }
      // Generate reshape Input 2D(Input_shape_i0 * Input_shape_i1 * ... Input_shape_ix-1, Input_shape_ix)
      SetFlatternNdLinearTensorStorageInfo(input_kernel_tensor_, input_reshape_size, x_shape);
      // Generate reshape Output 2D(Input_shape_i0 * Input_shape_i1 * ... Input_shape_ix-1, W_t_shape_last)
      SetFlatternNdLinearTensorStorageInfo(output_kernel_tensor_, input_reshape_size, w_t_shape);

      GetWorkspaceForResizeAddmm(inputs[kIndex2], input_kernel_tensor_.get(), w_t_, one_, one_,
                                 output_kernel_tensor_.get(), cube_math_type_);
    } else {
      auto matmul_res_ = &matmul_tensor_;
      matmul_tensor_.SetType(inputs[kIndex0]->GetType());
      ShapeVector matmul_shape = x_shape;
      matmul_shape[x_rank - 1] = w_t_shape[w_rank - 1];
      // When x and w are 1D, a scalar tensor(0D) is generated.
      if (x_rank == 1 && w_rank == 1) {
        matmul_shape = {};
      }
      matmul_tensor_.SetShape(std::make_shared<abstract::TensorShape>(matmul_shape));

      GetWorkspaceForResizeMatmul(inputs[kIndex0], w_t_, &matmul_tensor_, cube_math_type_);
      GetWorkspaceForResizeAdd(matmul_res_, bias, one_, outputs[kIndex0]);

      const auto &matmul_output_size =
        ops::CalOutputSize(matmul_tensor_.GetShapeVector(), mindspore::abstract::TypeIdSize(matmul_tensor_.dtype_id()));
      workspace_size_list_.emplace_back(matmul_output_size);
    }
  }
  const auto &w_t_output_size =
    ops::CalOutputSize(w_t_tensor_.GetShapeVector(), mindspore::abstract::TypeIdSize(w_t_tensor_.dtype_id()));

  workspace_size_list_.emplace_back(w_t_output_size);
}

bool DenseAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  auto x_rank = x_shape.size();
  // 1. Generate Transpose weight, the weight'rank greater than 0
  // The w_t_tensor always the last one
  size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(kIndex1));
  w_t_tensor_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto w_t_ = &w_t_tensor_;

  RunOpTransform(stream_ptr, workspace, inputs[kIndex1], t_perm_, &w_t_tensor_);
  // 2. Go through Different scenarios based on input'rank and bias'rank.
  auto bias = inputs[kIndex2];
  if (bias->GetType()->type_id() == kMetaTypeNone) {
    RunOpMatmul(stream_ptr, workspace, inputs[kIndex0], w_t_, outputs[kIndex0], cube_math_type_);
  } else {
    auto bias_shape = bias->GetShapeVector();
    auto bias_rank = bias_shape.size();
    if (x_rank == kDim2) {
      RunOpAddmm(stream_ptr, workspace, inputs[kIndex2], inputs[kIndex0], w_t_, one_, one_, outputs[kIndex0],
                 cube_math_type_);
    } else if (bias_rank == kDim1 || x_rank == kDim3) {
      input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
      output_kernel_tensor_->set_device_ptr(outputs[kIndex0]->device_ptr());

      RunOpAddmm(stream_ptr, workspace, inputs[kIndex2], input_kernel_tensor_.get(), w_t_, one_, one_,
                 output_kernel_tensor_.get(), cube_math_type_);
    } else {
      // If matmul_tensor_ has value, it's the penultimate one.
      matmul_tensor_.set_device_ptr(workspace[workspace_offset - kIndex1]->device_ptr());
      auto matmul_res_ = &matmul_tensor_;

      RunOpMatmul(stream_ptr, workspace, inputs[kIndex0], w_t_, &matmul_tensor_, cube_math_type_);
      RunOpAdd(stream_ptr, workspace, matmul_res_, bias, one_, outputs[kIndex0]);
    }
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Dense, DenseAclnnKernelMod);
}  // namespace dense
}  // namespace kernel
}  // namespace mindspore
