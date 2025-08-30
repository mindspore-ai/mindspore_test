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

#include "kernel/ascend/internal/add_layer_norm.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalAddLayerNorm::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                           const internal::OutputsImmutableInfoList &outputs_ii,
                                                           const std::vector<KernelTensor *> &ms_inputs,
                                                           const std::vector<KernelTensor *> &ms_outputs) {
  internal::NormParam param;
  param.eps = ms_inputs[kIndex4]->GetValueWithCheck<float>();

  MS_LOG(INFO) << "Create kernel: " << internal::kInternalAddLayerNormOpName << " eps: " << param.eps;
  return internal::CreateAddLayerNormOp(inputs_ii, outputs_ii, param, internal::kInternalAddLayerNormOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(AddLayerNorm, internal::kInternalAddLayerNormOpName, InternalAddLayerNorm);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(AddLayerNorm, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(AddLayerNorm, OUTPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
