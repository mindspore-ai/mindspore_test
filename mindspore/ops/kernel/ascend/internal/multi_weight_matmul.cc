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

#include "kernel/ascend/internal/multi_weight_matmul.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalMultiWeightMatmulBase::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                                    const internal::OutputsImmutableInfoList &outputs,
                                                                    const std::vector<KernelTensor *> &ms_inputs,
                                                                    const std::vector<KernelTensor *> &ms_outputs) {
  internal::MultiWeightMatmulParam param;
  const size_t kSizeNum2 = 2;
  auto n_lens = primitive_->GetAttr("n_lens");
  MS_EXCEPTION_IF_NULL(n_lens);
  auto n_list = GetValue<std::vector<int64_t>>(n_lens);
  if (n_list.size() == kSizeNum2) {
    n_list.push_back(0);
  }
  MS_LOG(INFO) << "Creating kernel for op_name " << op_name_;
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
  param.split_two = split_two_;
  param.fused = fused_;
  return internal::CreateMultiWeightMatmulOp(inputs, outputs, param, op_name_);
}

// MatmulSplitOut3
class InternalMatmulSplitOut3 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulSplitOut3() : InternalMultiWeightMatmulBase("InternalMatmulSplitOut3") {}
  ~InternalMatmulSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitOut3, internal::kInternalMultiWeightMatmulOpName, InternalMatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut3, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulSplitOut2
class InternalMatmulSplitOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulSplitOut2() : InternalMultiWeightMatmulBase("InternalMatmulSplitOut2") {}
  ~InternalMatmulSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitOut2, internal::kInternalMultiWeightMatmulOpName, InternalMatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantbatchmatmulSplitOut3
class InternalQuantbatchmatmulSplitOut3 : public InternalMultiWeightMatmulBase {
 public:
  InternalQuantbatchmatmulSplitOut3() : InternalMultiWeightMatmulBase("InternalQuantbatchmatmulSplitOut3") {}
  ~InternalQuantbatchmatmulSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantbatchmatmulSplitOut3, internal::kInternalMultiWeightMatmulOpName,
                               InternalQuantbatchmatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut3, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// QuantbatchmatmulSplitOut2
class InternalQuantbatchmatmulSplitOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalQuantbatchmatmulSplitOut2() : InternalMultiWeightMatmulBase("InternalQuantbatchmatmulSplitOut2") {}
  ~InternalQuantbatchmatmulSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantbatchmatmulSplitOut2, internal::kInternalMultiWeightMatmulOpName,
                               InternalQuantbatchmatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitOut3
class InternalMatmulBiasSplitOut3 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulBiasSplitOut3() : InternalMultiWeightMatmulBase("InternalMatmulBiasSplitOut3") {}
  ~InternalMatmulBiasSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitOut3, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulBiasSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut3, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulBiasSplitOut2
class InternalMatmulBiasSplitOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulBiasSplitOut2() : InternalMultiWeightMatmulBase("InternalMatmulBiasSplitOut2") {}
  ~InternalMatmulBiasSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitOut2, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulBiasSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulSplitSiluOut2
class InternalMatmulSplitSiluOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulSplitSiluOut2() : InternalMultiWeightMatmulBase("InternalMatmulSplitSiluOut2") {}
  ~InternalMatmulSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitSiluOut2, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitSiluOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantbatchmatmulSplitSiluOut2
class InternalQuantbatchmatmulSplitSiluOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalQuantbatchmatmulSplitSiluOut2() : InternalMultiWeightMatmulBase("InternalQuantbatchmatmulSplitSiluOut2") {}
  ~InternalQuantbatchmatmulSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantbatchmatmulSplitSiluOut2, internal::kInternalMultiWeightMatmulOpName,
                               InternalQuantbatchmatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantbatchmatmulSplitSiluOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantbatchmatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitSiluOut2
class InternalMatmulBiasSplitSiluOut2 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulBiasSplitSiluOut2() : InternalMultiWeightMatmulBase("InternalMatmulBiasSplitSiluOut2") {}
  ~InternalMatmulBiasSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitSiluOut2, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulBiasSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulSplitSiluMulOut1
class InternalMatmulSplitSiluMulOut1 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulSplitSiluMulOut1() : InternalMultiWeightMatmulBase("InternalMatmulSplitSiluMulOut1") {
    fused_ = true;
    split_two_ = true;
  }
  ~InternalMatmulSplitSiluMulOut1() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitSiluMulOut1, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulSplitSiluMulOut1);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitSiluMulOut1, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitSiluMulOut1, OUTPUT_NUM_1, INDEX_0);

// MatmulSplitSiluFastgeluAddMulOut1
class InternalMatmulSplitSiluFastgeluAddMulOut1 : public InternalMultiWeightMatmulBase {
 public:
  InternalMatmulSplitSiluFastgeluAddMulOut1()
      : InternalMultiWeightMatmulBase("InternalMatmulSplitSiluFastgeluAddMulOut1") {
    split_two_ = false;
    fused_ = true;
  }
  ~InternalMatmulSplitSiluFastgeluAddMulOut1() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitSiluFastgeluAddMulOut1, internal::kInternalMultiWeightMatmulOpName,
                               InternalMatmulSplitSiluFastgeluAddMulOut1);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitSiluFastgeluAddMulOut1, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitSiluFastgeluAddMulOut1, OUTPUT_NUM_1, INDEX_0);

// QMatmulSplitSiluMulOut1
class InternalQMatmulSplitSiluMulOut1 : public InternalMultiWeightMatmulBase {
 public:
  InternalQMatmulSplitSiluMulOut1() : InternalMultiWeightMatmulBase("InternalQMatmulSplitSiluMulOut1") {
    split_two_ = true;
    fused_ = true;
  }
  ~InternalQMatmulSplitSiluMulOut1() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QMatmulSplitSiluMulOut1, internal::kInternalMultiWeightMatmulOpName,
                               InternalQMatmulSplitSiluMulOut1);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QMatmulSplitSiluMulOut1, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QMatmulSplitSiluMulOut1, OUTPUT_NUM_1, INDEX_0);

// QMatmulSplitSiluFastgeluAddMulOut1
class InternalQMatmulSplitSiluFastgeluAddMulOut1 : public InternalMultiWeightMatmulBase {
 public:
  InternalQMatmulSplitSiluFastgeluAddMulOut1()
      : InternalMultiWeightMatmulBase("InternalQMatmulSplitSiluFastgeluAddMulOut1") {
    split_two_ = false;
    fused_ = true;
  }
  ~InternalQMatmulSplitSiluFastgeluAddMulOut1() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QMatmulSplitSiluFastgeluAddMulOut1, internal::kInternalMultiWeightMatmulOpName,
                               InternalQMatmulSplitSiluFastgeluAddMulOut1);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QMatmulSplitSiluFastgeluAddMulOut1, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2,
                                     INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QMatmulSplitSiluFastgeluAddMulOut1, OUTPUT_NUM_1, INDEX_0);

}  // namespace kernel
}  // namespace mindspore
