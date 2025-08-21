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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/count_nonzero_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace count_nonzero {

void CountNonZeroAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();

  TypeId other_type = inputs[kIndex0]->dtype_id();
  if (other_type == kNumberTypeComplex || other_type == kNumberTypeComplex64 || other_type == kNumberTypeComplex128) {
    MAKE_SCALAR(0, kNumberTypeInt8, other_);
  } else {
    MAKE_SCALAR(0, other_type, other_);
  }
  ne_tensor_.SetType(std::make_shared<TensorType>(kBool));
  ShapeVector shape = inputs[kIndex0]->GetShapeVector();
  ne_tensor_.SetShape(std::make_shared<abstract::TensorShape>(shape));
  const auto axis_opt = inputs[kIndex1]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (axis_opt.has_value()) {
    axis_ = axis_opt.value();
  } else {
    axis_ = std::vector<int64_t>{};
  }
  auto input = &ne_tensor_;
  GetWorkspaceForResizeNeScalar(inputs[kIndex0], other_, &ne_tensor_);
  GetWorkspaceForResizeReduceSum(input, axis_, keep_dims_, out_dtype_, outputs[kIndex0]);
  const auto &output_size =
    ops::CalOutputSize(ne_tensor_.GetShapeVector(), mindspore::abstract::TypeIdSize(ne_tensor_.dtype_id()));
  workspace_size_list_.emplace_back(output_size);
}

bool CountNonZeroAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(1));
  ne_tensor_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  auto input = &ne_tensor_;
  RunOpNeScalar(stream_ptr, workspace, inputs[kIndex0], other_, &ne_tensor_);
  RunOpReduceSum(stream_ptr, workspace, input, axis_, keep_dims_, out_dtype_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(CountNonZero, CountNonZeroAscend);
}  // namespace count_nonzero
}  // namespace kernel
}  // namespace mindspore
