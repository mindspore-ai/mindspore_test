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

#include "kernel/ascend/internal/swiglu_dynamic_quant.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto swiglu_fusion_str = "swiglu_v2";
internal::InternalOpPtr InternalSwiGLUDynamicQuant::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                 const internal::OutputsImmutableInfoList &outputs_ii,
                                                                 const std::vector<KernelTensor *> &ms_inputs,
                                                                 const std::vector<KernelTensor *> &ms_outputs) {
  internal::SwiGLUParam param;
  param.axis = -1;
  param.is_fusion_v2 = false;
  param.with_dyn_quant = true;
  auto value_str = primitive_->GetAttr("FusionType");
  MS_EXCEPTION_IF_NULL(value_str);
  std::string fusion_type = GetValue<std::string>(value_str);
  if (fusion_type == swiglu_fusion_str) {
    param.is_fusion_v2 = true;
  }
  return internal::CreateSwiGLUDynamicQuantOp(inputs_ii, outputs_ii, param,
                                              internal::kInternalSwiGLUDynamicQuantOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(SwiGLUDynamicQuant, internal::kInternalSwiGLUDynamicQuantOpName,
                               InternalSwiGLUDynamicQuant);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(SwiGLUDynamicQuant, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(SwiGLUDynamicQuant, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
