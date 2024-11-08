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
#include "backend/common/graph_kernel/expander/base/ir_builder.h"
#include "backend/common/graph_kernel/expander/base/utils.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("AddN").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib, FormatDefaultNchwSame)) {
    return {};
  }
  // Check Inputs
  constexpr size_t min_inputs = 2;
  if (ib->inputs().size() < min_inputs) {
    MS_LOG(INFO) << "For 'AddN', the inputs num should be greater than 1, but got " << ib->inputs().size();
    return {};
  }

  auto result = ib->input(0);
  for (size_t i = 1; i < ib->inputs().size(); ++i) {
    result = ib->Add(result, ib->input(i));
  }
  return {result};
});

REG_EXPANDER_FUNC("AssignAdd").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  auto param = ib->input(0);
  auto x = ib->input(1);
  if (x->GetDtype() != param->GetDtype()) {
    x = ib->Cast(x, param->GetDtype());
  }
  auto result = ib->Assign(param, ib->Add(param, x));
  return {result};
});

REG_EXPANDER_FUNC("Tanh").SetBody(BODYFUNC(ib) {
  auto result = ib->Tanh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Sinh").SetBody(BODYFUNC(ib) {
  auto result = ib->Sinh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Cosh").SetBody(BODYFUNC(ib) {
  auto result = ib->Cosh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("LogicalXor").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  const auto &input_y = ib->input(kIndex1);

  auto result_b = ib->LogicalAnd(input_x, ib->LogicalNot(input_y));
  auto result_a = ib->LogicalAnd(input_y, ib->LogicalNot(input_x));
  return {ib->LogicalOr(result_a, result_b)};
});

NodePtrList FastGeluExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(-val, input_x->GetDtype());
  auto const_1 = ib->Tensor(val / 2, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto sub = ib->Sub(input_x, abs);
  auto exp_0 = ib->Exp(ib->Mul(const_1, sub));
  auto n = ib->Mul(input_x, exp_0);
  auto exp_1 = ib->Exp(ib->Mul(const_0, abs));
  auto d = ib->Add(exp_1, const_2);

  return {ib->Div(n, d)};
}

NodePtrList FastGeluGradExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex1);
  const auto &dout = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(val, input_x->GetDtype());
  auto const_1 = ib->Tensor(-val, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto mul_1 = ib->Exp(ib->Mul(const_1, abs));
  auto mul_3 = ib->Mul(input_x, mul_1);
  mul_3 = ib->Mul(const_0, mul_3);
  mul_3 = ib->Add(mul_3, mul_1);

  auto sub = ib->Sub(input_x, abs);
  sub = ib->Exp(ib->Mul(sub, const_0));

  mul_3 = ib->Add(sub, mul_3);
  mul_1 = ib->Add(mul_1, const_2);
  mul_1 = ib->Mul(mul_1, mul_1);

  return {ib->Mul(ib->Div(mul_3, mul_1), dout)};
}

REG_EXPANDER_FUNC("FastGelu").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeLU").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeluGrad").SetBody(FastGeluGradExpand);
REG_EXPANDER_FUNC("FastGeLUGrad").SetBody(FastGeluGradExpand);

REG_EXPANDER_FUNC("SiLU").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto result = ib->Div(input_x, add_exp);
  return {result};
});

REG_EXPANDER_FUNC("SiLUGrad").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex1);
  const auto &dout = ib->input(kIndex0);
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto sigmod = ib->Div(const_one, add_exp);
  auto out = ib->Div(input_x, add_exp);

  auto result = ib->Sub(ib->Add(sigmod, out), ib->Mul(sigmod, out));
  return {ib->Mul(result, dout)};
});

REG_EXPANDER_FUNC("Addcmul").SetBody(BODYFUNC(ib) {
  const auto &input_data = ib->input(kIndex0);
  const auto &x1 = ib->input(kIndex1);
  const auto &x2 = ib->input(kIndex2);
  const auto &value = ib->input(kIndex3);
  auto result = ib->Add(input_data, ib->Mul(value, ib->Mul(x1, x2)));
  return {result};
});

REG_EXPANDER_FUNC("ReluGrad").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib)) {
    return {};
  }
  const auto &input_dout = ib->input(kIndex0);
  const auto &input_x = ib->input(kIndex1);
  auto const_zero = ib->Tensor(0.0, input_x->GetDtype());
  auto gt_res = ib->Greater(input_x, const_zero);
  auto res = ib->Select(gt_res, input_dout, const_zero);
  return {res};
});
}  // namespace mindspore::graphkernel::expander
