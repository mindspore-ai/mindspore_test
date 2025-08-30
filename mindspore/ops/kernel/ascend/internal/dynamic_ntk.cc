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

#include "kernel/ascend/internal/dynamic_ntk.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {

enum class DynamicNTKOutType { Float16 = 0, BFloat16 = 1, Float32 = 2 };

internal::InternalOpPtr InternalDynamicNTK::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                         const internal::OutputsImmutableInfoList &outputs_ii,
                                                         const std::vector<KernelTensor *> &ms_inputs,
                                                         const std::vector<KernelTensor *> &ms_outputs) {
  internal::DynamicNTKParam param;
  int64_t out_type = ms_inputs[kIndex3]->GetValueWithCheck<int64_t>();
  TypeId out_dtype = static_cast<TypeId>(out_type);

  if (out_dtype == TypeId::kNumberTypeFloat16) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::Float16);
  } else if (out_dtype == TypeId::kNumberTypeBFloat16) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::BFloat16);
  } else if (out_dtype == TypeId::kNumberTypeFloat32) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::Float32);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported output type: " << out_dtype;
  }
  return internal::CreateDynamicNTKOp(inputs_ii, outputs_ii, param, internal::kInternalDynamicNTKOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(DynamicNTK, internal::kInternalDynamicNTKOpName, InternalDynamicNTK);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(DynamicNTK, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(DynamicNTK, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
