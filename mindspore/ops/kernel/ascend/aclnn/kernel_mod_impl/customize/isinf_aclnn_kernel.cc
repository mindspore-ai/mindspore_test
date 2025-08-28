/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/isinf_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include <limits>

#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "mindspore/core/include/base/float16.h"

namespace mindspore {
namespace kernel {
namespace isinf {

void IsInfAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();

  auto type_id = inputs[kIndex0]->dtype_id();
  is_int_val_ = (type_id >= kNumberTypeBool) && (type_id < kNumberTypeFloat);
  if (is_int_val_) {
    GetWorkspaceForResizeInplaceZero(outputs[kIndex0]);
    return;
  }

  abs_out_.SetType(inputs[kIndex0]->GetType());
  abs_out_.SetShape(std::make_shared<abstract::TensorShape>(input_shape));
  GetWorkspaceForResizeAbs(inputs[kIndex0], &abs_out_);

  switch (type_id) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32: {
      MAKE_SCALAR(std::numeric_limits<float>::infinity(), type_id, inf_ptr_);
      GetWorkspaceForResizeEqScalar(&abs_out_, inf_ptr_, outputs[kIndex0]);
      break;
    }
    case kNumberTypeFloat64: {
      MAKE_SCALAR(std::numeric_limits<double>::infinity(), type_id, inf_ptr_);
      GetWorkspaceForResizeEqScalar(&abs_out_, inf_ptr_, outputs[kIndex0]);
      break;
    }
    case kNumberTypeFloat16: {
      MAKE_SCALAR(std::numeric_limits<Float16>::infinity(), type_id, inf_ptr_);
      GetWorkspaceForResizeEqScalar(&abs_out_, inf_ptr_, outputs[kIndex0]);
      break;
    }
    case kNumberTypeBFloat16: {
      MAKE_SCALAR(std::numeric_limits<BFloat16>::infinity(), type_id, inf_ptr_);
      GetWorkspaceForResizeEqScalar(&abs_out_, inf_ptr_, outputs[kIndex0]);
      break;
    }
    default:
      MS_EXCEPTION(TypeError) << "Op IsInf does not support type " << TypeIdToString(type_id);
      break;
  }

  const auto &output_size =
    ops::CalOutputSize(abs_out_.GetShapeVector(), mindspore::abstract::TypeIdSize(abs_out_.dtype_id()));
  workspace_size_list_.emplace_back(output_size);
}

bool IsInfAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  if (is_int_val_) {
    RunOpInplaceZero(stream_ptr, workspace, outputs[kIndex0]);
    return true;
  }

  size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(1));
  abs_out_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  RunOpAbs(stream_ptr, workspace, inputs[kIndex0], &abs_out_);

  RunOpEqScalar(stream_ptr, workspace, &abs_out_, inf_ptr_, outputs[kIndex0]);

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(IsInf, IsInfAscend);
}  // namespace isinf
}  // namespace kernel
}  // namespace mindspore
