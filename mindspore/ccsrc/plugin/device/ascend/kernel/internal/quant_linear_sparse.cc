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

#include "plugin/device/ascend/kernel/internal/quant_linear_sparse.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalQuantLinearSparse::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  internal::MatMulParam matmul_param;
  param_ptr->opId = internal::OpId::QuantLinearSparse;

  auto shape_x = inputs[kIndex0]->GetShapeVector();
  auto shape_deqScale = inputs[kIndex2]->GetShapeVector();
  int m = shape_x[kIndex0];
  int k = shape_x[kIndex1];
  int n = shape_deqScale[shape_deqScale.size() - 1];

  matmul_param = {
    false,      // transposeA
    true,       // transposeB
    {m, k, n},  // oriShape
    true,       // withBias
    true,       // enDequant
    8,          // tilingN
    8,          // tilingK
  };
  param_ptr->specificParam = matmul_param;
  return std::static_pointer_cast<internal::OpParam>(param_ptr);
}

MS_INTERNAL_KERNEL_FACTORY_REG(QuantLinearSparse, InternalQuantLinearSparse);
}  // namespace kernel
}  // namespace mindspore
