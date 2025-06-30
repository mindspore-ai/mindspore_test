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

#include "plugin/device/ascend/kernel/internal/gather_pre_rms_norm.h"

#include <memory>
#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalGatherPreRmsNorm::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                               const internal::OutputsImmutableInfoList &outputs_ii,
                                                               const std::vector<KernelTensor *> &ms_inputs,
                                                               const std::vector<KernelTensor *> &ms_outputs) {
  internal::NormParam param;
  param.eps = ms_inputs[kIndex4]->GetValueWithCheck<float>();

  MS_LOG(INFO) << "Create kernel: " << internal::kInternalGatherPreRmsNormOpName << " eps: " << param.eps;
  return internal::CreateGatherPreRmsNormOp(inputs_ii, outputs_ii, param, internal::kInternalGatherPreRmsNormOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(GatherPreRmsNorm, internal::kInternalGatherPreRmsNormOpName, InternalGatherPreRmsNorm);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(GatherPreRmsNorm, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(GatherPreRmsNorm, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
