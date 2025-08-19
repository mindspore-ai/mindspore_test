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
#include <algorithm>
#include <vector>
#include <string>
#include "ir/tensor.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/matmul_allreduce_add_rmsnorm_aclnn_kernel.h"

namespace mindspore {
namespace kernel {
namespace matmul_allreduce_add_rmsnorm {
void MatmulAllReduceAddRmsNormAscend::InitInputAttributes(const std::vector<KernelTensor *> &inputs,
                                                          const std::vector<KernelTensor *> &outputs) {
  auto eps_dtype_id = inputs[kIndex5]->dtype_id();
  eps_ = (eps_dtype_id == kNumberTypeFloat32) ? static_cast<double>(inputs[kIndex5]->GetValueWithCheck<float>())
                                              : inputs[kIndex5]->GetValueWithCheck<double>();
  auto group = inputs[kIndex6]->GetValueWithCheck<std::string>();
  comm_name_ = OpApiUtil::GetCommName(group);
  auto reduction_enum = inputs[kIndex7]->GetValueWithCheck<int64_t>();
  reduce_op_ = device::ascend::GEReduction::ConvertEnumToString(reduction_enum);
  comm_turn_ = inputs[kIndex8]->GetValueWithCheck<int64_t>();
  stream_mode_ = inputs[kIndex9]->GetValueWithCheck<int64_t>();

  // check whether bias is empty tensor
  auto bias_shape_vector = inputs[kIndex2]->GetShapeVector();
  is_bias_empty_ =
    std::any_of(bias_shape_vector.begin(), bias_shape_vector.end(), [](const int64_t &dim) { return dim == 0; });
}

void MatmulAllReduceAddRmsNormAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  InitInputAttributes(inputs, outputs);
  if (is_bias_empty_) {
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], nullptr, inputs[kIndex3], inputs[kIndex4], eps_, comm_name_,
                          reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0], outputs[kIndex1]);
  } else {
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], eps_,
                          comm_name_, reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0], outputs[kIndex1]);
  }
}

bool MatmulAllReduceAddRmsNormAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (is_bias_empty_) {
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], nullptr, inputs[kIndex3], inputs[kIndex4], eps_,
          comm_name_, reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0], outputs[kIndex1]);
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
          eps_, comm_name_, reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0], outputs[kIndex1]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MatmulAllReduceAddRmsNorm, MatmulAllReduceAddRmsNormAscend);
}  // namespace matmul_allreduce_add_rmsnorm
}  // namespace kernel
}  // namespace mindspore
