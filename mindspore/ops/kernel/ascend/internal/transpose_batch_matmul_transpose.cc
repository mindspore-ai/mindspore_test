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

#include "kernel/ascend/internal/transpose_batch_matmul_transpose.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalTransposeBatchMatmulTranspose::CreateKernel(
  const internal::InputsImmutableInfoList &inputs_ii, const internal::OutputsImmutableInfoList &outputs_ii,
  const std::vector<KernelTensor *> &ms_inputs, const std::vector<KernelTensor *> &ms_outputs) {
  internal::TransBMMTransParam param;
  param.perm_in = ms_inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  param.perm_out = ms_inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  param.transpose_a = ms_inputs[kIndex4]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[kIndex5]->GetValueWithCheck<bool>();
  if (param.perm_in.size() != param.perm_out.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', size of perm_in should equal to the perm_out, but got "
                      << param.perm_in.size() << " and " << param.perm_out.size();
  }
  auto shape = ms_inputs[kIndex0]->GetShapeVector();
  for (size_t i = 0; i < param.perm_in.size(); ++i) {
    if (param.perm_in[i] < 0) {
      param.perm_in[i] += shape.size();
    }

    if (param.perm_out[i] < 0) {
      param.perm_out[i] += shape.size();
    }
  }
  return internal::CreateTransposeBatchMatmulTransposeOp(inputs_ii, outputs_ii, param,
                                                         internal::kInternalTransBMMTransOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(TransposeBatchMatmulTranspose, internal::kInternalTransBMMTransOpName,
                               InternalTransposeBatchMatmulTranspose);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(TransposeBatchMatmulTranspose, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(TransposeBatchMatmulTranspose, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
