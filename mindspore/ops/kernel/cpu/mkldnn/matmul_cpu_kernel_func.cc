/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/mkldnn/matmul_cpu_kernel_func.h"
#include <utility>
#include <map>
#include <memory>
#include <string>

#include "mindspore/ops/infer/ops_func_impl/matmul.h"
#include "include/common/utils/utils.h"
#include "common/common_utils.h"
#include "mkldnn/mkl_cpu_kernel.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "ops/base_operator.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/math_op_name.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 4;
constexpr size_t kMatMulWithBiasAddInputsNum = 5;
constexpr size_t kBiasAddInputIndex = kMatMulWithBiasAddInputsNum - 3;
constexpr size_t kMatMulOutputsNum = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kRankMin = 2;
using dims = dnnl::memory::dims;
}  // namespace

void MatMulCpuKernelFunc::InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  prim_ = primitive;
}

void MatMulCpuKernelFunc::ProcessTranspose(const std::vector<KernelTensor *> &inputs) {
  auto input_size = inputs.size();
  if (prim_->name() == kBatchMatMulExtOpName) {
    trans_a_ = false;
    trans_b_ = false;
  } else {
    auto transpose_x1_opt = inputs[input_size - 2]->GetOptionalValueWithCheck<bool>();
    auto transpose_x2_opt = inputs[input_size - 1]->GetOptionalValueWithCheck<bool>();
    if (!transpose_x1_opt.has_value() || !transpose_x2_opt.has_value()) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', transpose_a and transpose_b should be specified.";
    }
    trans_a_ = transpose_x1_opt.value();
    trans_b_ = transpose_x2_opt.value();
  }
}
int MatMulCpuKernelFunc::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  with_relu_ = false;
  if (prim_->GetAttr(kAttrWithRelu) != nullptr) {
    with_relu_ = GetValue<bool>(prim_->GetAttr(kAttrWithRelu));
  }
  MatMulCpuKernelFunc::ProcessTranspose(inputs);
  auto a_shape = inputs[kIndex0]->GetShapeVector();
  auto b_shape = inputs[kIndex1]->GetShapeVector();
  with_bias_add_ = false;
  if (prim_->GetAttr(kAttrWithBiasAdd) != nullptr) {
    with_bias_add_ = GetValue<bool>(prim_->GetAttr(kAttrWithBiasAdd));
  }
  auto o_shape = outputs[kIndex0]->GetShapeVector();
  bool flag = a_shape.size() < kRankMin || b_shape.size() < kRankMin || o_shape.size() < kRankMin;
  if (flag) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul must be greater than or equal to " << kRankMin;
  }
  auto rank = a_shape.size();
  int64_t batch = 1;
  for (size_t i = 0; i < rank - kIndexOffset; ++i) {
    batch *= a_shape[i];
  }

  int64_t dim_m = o_shape[rank - kIndexOffset];
  int64_t dim_n = o_shape[rank - 1];
  int64_t dim_k = 1;
  if (trans_a_) {
    dim_k = a_shape[rank - kIndexOffset];
  } else {
    dim_k = a_shape[rank - 1];
  }

  dims src_dims;
  dims weights_dims;
  dims bias_dims;
  dims dst_dims;
  dims a_strides;
  dims b_strides;
  dims o_strides;
  dims bias_strides;
  if (batch > 1) {
    src_dims = {batch, dim_m, dim_k};
    weights_dims = {batch, dim_k, dim_n};
    dst_dims = {batch, dim_m, dim_n};
    a_strides = {trans_a_ ? dims{dim_m * dim_k, 1, dim_m} : dims{dim_m * dim_k, dim_k, 1}};
    b_strides = {trans_b_ ? dims{dim_n * dim_k, 1, dim_k} : dims{dim_n * dim_k, dim_n, 1}};
    o_strides = {dim_n * dim_m, dim_n, 1};
    if (with_bias_add_) {
      bias_dims = {1, 1, dim_n};
      bias_strides = dims{dim_n, dim_n, 1};
    }
  } else {
    src_dims = {dim_m, dim_k};
    weights_dims = {dim_k, dim_n};
    dst_dims = {dim_m, dim_n};
    a_strides = {trans_a_ ? dims{1, dim_m} : dims{dim_k, 1}};
    b_strides = {trans_b_ ? dims{1, dim_k} : dims{dim_n, 1}};
    o_strides = {dim_n, 1};
    if (with_bias_add_) {
      bias_dims = {1, dim_n};
      bias_strides = dims{dim_n, 1};
    }
  }

  auto src_md = CreateDesc<dnnl::memory::desc>(src_dims, dnnl::memory::data_type::f32, a_strides);
  auto weights_md = CreateDesc<dnnl::memory::desc>(weights_dims, dnnl::memory::data_type::f32, b_strides);
  auto dst_md = CreateDesc<dnnl::memory::desc>(dst_dims, dnnl::memory::data_type::f32, o_strides);
  auto matmul_desc = CreateDesc<dnnl::matmul::desc>(src_md, weights_md, dst_md);

  if (with_bias_add_) {
    auto bias_md = CreateDesc<dnnl::memory::desc>(bias_dims, dnnl::memory::data_type::f32, bias_strides);
    matmul_desc = CreateDesc<dnnl::matmul::desc>(src_md, weights_md, bias_md, dst_md);
    AddArgument(DNNL_ARG_BIAS, bias_md);
  }

  auto prim_desc = CreateDesc<dnnl::matmul::primitive_desc>(matmul_desc, engine_);

  if (with_relu_) {
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    dnnl::post_ops matmul_ops;
    matmul_ops.append_eltwise(scale, dnnl::algorithm::eltwise_relu, alpha, beta);
    dnnl::primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    prim_desc = CreateDesc<dnnl::matmul::primitive_desc>(matmul_desc, matmul_attr, engine_);
  }

  primitive_ = CreatePrimitive<dnnl::matmul>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_md);
  AddArgument(DNNL_ARG_WEIGHTS, weights_md);
  AddArgument(DNNL_ARG_DST, dst_md);

  return KRET_OK;
}

bool MatMulCpuKernelFunc::RunFunc(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  if (with_bias_add_) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulWithBiasAddInputsNum, kernel_name_);
    SetArgumentHandle(DNNL_ARG_BIAS, GetDeviceAddress<float>(inputs, kBiasAddInputIndex));
  } else if (prim_->name() == kBatchMatMulExtOpName) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum - kIndexOffset, kernel_name_);
  } else {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  }
  const auto input_a = GetDeviceAddress<float>(inputs, kIndex0);
  const auto input_b = GetDeviceAddress<float>(inputs, kIndex1);
  auto output = GetDeviceAddress<float>(outputs, kIndex0);

  SetArgumentHandle(DNNL_ARG_SRC, input_a);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, input_b);
  SetArgumentHandle(DNNL_ARG_DST, output);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
