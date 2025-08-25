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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_index_fill_tensor_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {

void InplaceIndexFillTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  auto index_shape = inputs[kIndex2]->GetShapeVector();
  auto value_shape = inputs[kIndex3]->GetShapeVector();
  if (MS_UNLIKELY(index_shape.size() > 1)) {
    MS_LOG(EXCEPTION) << "For [InplaceIndexFillTensor], the rank of input 'index'"
                      << " must be in [0, 1], but got " << index_shape.size() << ".";
  }
  if (MS_UNLIKELY(value_shape.size() != 0)) {
    MS_LOG(EXCEPTION) << "For [InplaceIndexFillTensor], the rank of input 'value'"
                      << " must be equal 0, but got " << value_shape.size() << ".";
  }
  dim_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  value_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex3]);
  auto index_type_id = inputs[kIndex2]->dtype_id();
  if (index_type_id == kNumberTypeInt64) {
    auto index = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
    index_vector_.assign(index.begin(), index.end());
  } else if (index_type_id == kNumberTypeInt32) {
    auto index = device::ascend::ConvertKernelTensor<std::vector<int32_t>>(inputs[kIndex2]);
    index_vector_.assign(index.begin(), index.end());
  } else {
    MS_LOG(EXCEPTION) << "For [InplaceIndexFillTensor], the input 'index'"
                      << " for conversion to int array must be of type Int32 or Int64,"
                      << " but got " << TypeIdToString(index_type_id);
  }
  GetWorkspaceForResize(inputs[kIndex0], dim_, index_vector_, value_);
}

bool InplaceIndexFillTensorAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, index_vector_, value_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceIndexFillTensor, InplaceIndexFillTensorAscend);
}  // namespace kernel
}  // namespace mindspore
