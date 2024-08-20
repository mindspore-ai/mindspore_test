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

#include "plugin/device/ascend/kernel/internal/acme/multi_weight_matmul.h"

#include <memory>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeMultiWeightMatmulBase::CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                                        const acme::OutputsImmutableInfoList &outputs,
                                                        const std::vector<KernelTensor *> &ms_inputs,
                                                        const std::vector<KernelTensor *> &ms_outputs) {
  acme::MultiWeightMatmulParam param;
  const size_t kSizeNum2 = 2;
  auto n_lens = primitive_->GetAttr("n_lens");
  MS_EXCEPTION_IF_NULL(n_lens);
  auto n_list = GetValue<std::vector<int64_t>>(n_lens);
  if (n_list.size() == kSizeNum2) {
    n_list.push_back(0);
  }
  const auto n_input_zero = 0;
  const auto n_input_one = 1;
  const auto n_input_two = 2;
  param.n0_len = static_cast<uint32_t>(n_list[n_input_zero]);
  param.n1_len = static_cast<uint32_t>(n_list[n_input_one]);
  param.n2_len = static_cast<uint32_t>(n_list[n_input_two]);
  bool with_bias = primitive_->HasAttr("with_bias") ? GetValue<bool>(primitive_->GetAttr("with_bias")) : false;
  int32_t silu_position =
    primitive_->HasAttr("silu_position") ? GetValue<int32_t>(primitive_->GetAttr("silu_position")) : -1;
  param.silu_position = silu_position;
  param.with_bias = with_bias;
  param.transpose_a = false;
  param.transpose_b = true;
  return acme::CreateMultiWeightMatmulOp(inputs, outputs, param, op_name_);
}

// MatmulSplitOut3
class AcmeMatmulSplitOut3 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulSplitOut3() : AcmeMultiWeightMatmulBase("AcmeMatmulSplitOut3") {}
  ~AcmeMatmulSplitOut3() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulSplitOut3, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut3, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulSplitOut2
class AcmeMatmulSplitOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulSplitOut2() : AcmeMultiWeightMatmulBase("AcmeMatmulSplitOut2") {}
  ~AcmeMatmulSplitOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulSplitOut2, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantbatchmatmulSplitOut3
class AcmeQuantbatchmatmulSplitOut3 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeQuantbatchmatmulSplitOut3() : AcmeMultiWeightMatmulBase("AcmeQuantbatchmatmulSplitOut3") {}
  ~AcmeQuantbatchmatmulSplitOut3() = default;
};

MS_ACME_KERNEL_FACTORY_REG(QuantbatchmatmulSplitOut3, acme::kAcmeMultiWeightMatmulOpName,
                           AcmeQuantbatchmatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut3, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// QuantbatchmatmulSplitOut2
class AcmeQuantbatchmatmulSplitOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeQuantbatchmatmulSplitOut2() : AcmeMultiWeightMatmulBase("AcmeQuantbatchmatmulSplitOut2") {}
  ~AcmeQuantbatchmatmulSplitOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(QuantbatchmatmulSplitOut2, acme::kAcmeMultiWeightMatmulOpName,
                           AcmeQuantbatchmatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitOut3
class AcmeMatmulBiasSplitOut3 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulBiasSplitOut3() : AcmeMultiWeightMatmulBase("AcmeMatmulBiasSplitOut3") {}
  ~AcmeMatmulBiasSplitOut3() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulBiasSplitOut3, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulBiasSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut3, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulBiasSplitOut2
class AcmeMatmulBiasSplitOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulBiasSplitOut2() : AcmeMultiWeightMatmulBase("AcmeMatmulBiasSplitOut2") {}
  ~AcmeMatmulBiasSplitOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulBiasSplitOut2, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulBiasSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulSplitSiluOut2
class AcmeMatmulSplitSiluOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulSplitSiluOut2() : AcmeMultiWeightMatmulBase("AcmeMatmulSplitSiluOut2") {}
  ~AcmeMatmulSplitSiluOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulSplitSiluOut2, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitSiluOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantbatchmatmulSplitSiluOut2
class AcmeQuantbatchmatmulSplitSiluOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeQuantbatchmatmulSplitSiluOut2() : AcmeMultiWeightMatmulBase("AcmeQuantbatchmatmulSplitSiluOut2") {}
  ~AcmeQuantbatchmatmulSplitSiluOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(QuantbatchmatmulSplitSiluOut2, acme::kAcmeMultiWeightMatmulOpName,
                           AcmeQuantbatchmatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitSiluOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitSiluOut2
class AcmeMatmulBiasSplitSiluOut2 : public AcmeMultiWeightMatmulBase {
 public:
  AcmeMatmulBiasSplitSiluOut2() : AcmeMultiWeightMatmulBase("AcmeMatmulBiasSplitSiluOut2") {}
  ~AcmeMatmulBiasSplitSiluOut2() = default;
};

MS_ACME_KERNEL_FACTORY_REG(MatmulBiasSplitSiluOut2, acme::kAcmeMultiWeightMatmulOpName, AcmeMatmulBiasSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
