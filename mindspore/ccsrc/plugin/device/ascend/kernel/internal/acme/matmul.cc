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

#include "plugin/device/ascend/kernel/internal/acme/matmul.h"

#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeMatmul::CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                         const acme::OutputsImmutableInfoList &outputs,
                                         const std::vector<KernelTensor *> &ms_inputs,
                                         const std::vector<KernelTensor *> &ms_outputs) {
  acme::MatmulParam param;
  auto input_len = ms_inputs.size();
  param.transpose_a = ms_inputs[input_len - kIndex2]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[input_len - kIndex1]->GetValueWithCheck<bool>();
  return acme::CreateMatmulOp(inputs, outputs, param, acme::kAcmeMatMulOpName);
}
MS_ACME_KERNEL_FACTORY_REG(MatMul, acme::kAcmeMatMulOpName, AcmeMatmul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatMul, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatMul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
