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

#include "kernel/ascend/internal/matmul_elemwise.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto matmul_elemwise_fusion_relu_str = "relu";
constexpr auto matmul_elemwise_fusion_gelu_str = "gelu";
constexpr auto matmul_elemwise_fusion_fastgelu_str = "fastgelu";
constexpr auto matmul_elemwise_fusion_biasadd_str = "bias_add";
constexpr auto matmul_elemwise_fusion_biasadd_fastgelu_str = "bias_add_fastgelu";
constexpr auto matmul_elemwise_fusion_sigmoid_add_str = "sigmoid_add";
}  // namespace

internal::InternalOpPtr InternalFusedMatmulElemBase::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                                  const internal::OutputsImmutableInfoList &outputs,
                                                                  const std::vector<KernelTensor *> &ms_inputs,
                                                                  const std::vector<KernelTensor *> &ms_outputs) {
  param_.transpose_a = primitive_->HasAttr("is_trans_a") ? GetValue<bool>(primitive_->GetAttr("is_trans_a")) : false;
  param_.transpose_b = primitive_->HasAttr("is_trans_b") ? GetValue<bool>(primitive_->GetAttr("is_trans_b")) : false;
  auto value_str = primitive_->GetAttr("ElemwiseType");
  MS_EXCEPTION_IF_NULL(value_str);
  std::string elemwise_type = GetValue<std::string>(value_str);
  if (elemwise_type == matmul_elemwise_fusion_relu_str) {
    param_.with_relu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_gelu_str) {
    param_.with_gelu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_biasadd_str) {
    param_.with_bias = true;
  } else if (elemwise_type == matmul_elemwise_fusion_biasadd_fastgelu_str) {
    param_.with_bias_fastgelu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_fastgelu_str) {
    param_.with_fastgelu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_sigmoid_add_str) {
    param_.with_sigmoid_add = true;
  }
  param_.enable_shuffle = false;  // the real definition is in internal
  param_.enable_dequant = false;
  return internal::CreateMatmulOp(inputs, outputs, param_, internal::kInternalMatMulOpName);
}

uint64_t InternalFusedMatmulElemBase::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_);
}

MS_INTERNAL_KERNEL_FACTORY_REG(FusedMatmulElemBinary, internal::kInternalMatMulOpName, InternalFusedMatmulElemBinary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatmulElemBinary, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatmulElemBinary, OUTPUT_NUM_1, INDEX_0);

MS_INTERNAL_KERNEL_FACTORY_REG(FusedMatmulElemUnary, internal::kInternalMatMulOpName, InternalFusedMatmulElemUnary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatmulElemUnary, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatmulElemUnary, OUTPUT_NUM_1, INDEX_0);

}  // namespace kernel
}  // namespace mindspore
