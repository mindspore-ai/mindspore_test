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

#include "plugin/device/ascend/kernel/internal/rms_norm.h"

#include <memory>

#include "common/kernel.h"
#include "mindspore/core/include/mindapi/base/types.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalRmsNorm::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                      const internal::OutputsImmutableInfoList &outputs_ii,
                                                      const std::vector<KernelTensor *> &ms_inputs,
                                                      const std::vector<KernelTensor *> &ms_outputs) {
  internal::NormParam norm_param;
  norm_param.eps = ms_inputs[kIndex2]->GetValueWithCheck<pyfloat>();
  return internal::CreateRmsNormOp(inputs_ii, outputs_ii, norm_param, internal::kInternalRmsNormOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(RmsNorm, internal::kInternalRmsNormOpName, InternalRmsNorm);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(RmsNorm, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(RmsNorm, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
