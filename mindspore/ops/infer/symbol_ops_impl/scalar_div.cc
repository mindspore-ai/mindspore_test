/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <utility>
#include "mindspore/ops/infer/symbol_ops_impl/scalar_cast.h"

namespace mindspore {
namespace symshape {
namespace ops {
std::pair<int64_t, int64_t> UpdateRange(int64_t min1, int64_t min2, int64_t max1, int64_t max2) {
  std::vector<int64_t> v;
  v.push_back(RangeDiv(min1, min2));
  v.push_back(RangeDiv(min1, max2));
  v.push_back(RangeDiv(max1, min2));
  v.push_back(RangeDiv(max1, max2));
  if (min2 <= -1 && -1 <= max2) {
    v.push_back(-min1);
    v.push_back(-max1);
  }
  if (min2 <= 1 && 1 <= max2) {
    v.push_back(min1);
    v.push_back(max1);
  }
  return std::make_pair(*std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
}

SymbolPtr ScalarDiv::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(DivWithCheck(lhs->value(), rhs->value()));
  }
  if (lhs->HasData() && lhs->value() == 0) {
    return GenInt(0);
  }
  if (rhs->HasData() && rhs->value() == 1) {
    DoNotEvalOnRun();
    return input(0);
  }
  if (*lhs == *rhs) {
    DoNotEvalOnRun();
    return GenInt(1);
  }
  return GenVInt();
}

void ScalarDiv::UpdateMathInfo() {
  if (!need_eval()) {
    return;
  }
  auto input1 = input_as_sptr<IntSymbol>(0);
  auto input2 = input_as_sptr<IntSymbol>(1);
  auto out = output_as<IntSymbol>();
  auto r = UpdateRange(input1->range_min(), input2->range_min(), input1->range_max(), input2->range_max());
  out->SetRange(r.first, r.second);
  // only support "s / const", does not support "const / s".
  if (input2->is_const() && !input1->is_const()) {
    // out = input1 / const2
    out->SetMathExpr(input1, Frac(1, input2->value()), 0);
  } else if (!input1->is_const() && !input2->is_const()) {
    auto d1 = input1->divisor();
    auto d2 = input2->divisor();
    auto r1 = input1->remainder();
    auto r2 = input2->remainder();
    if (r1 == 0 && r2 == 0 && d1 % d2 == 0) {
      out->SetDivisorRemainder(d1 / d2, 0);
    }
  }
}

SymbolPtr ScalarFloorDiv::Eval() {
  // only eval on Building
  auto lhs = input_as_sptr<IntSymbol>(0);
  auto rhs = input_as_sptr<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(FloorDiv(lhs->value(), rhs->value()));
  }
  if (lhs->is_divisible_by(rhs)) {
    DoNotEvalOnRun();
    return Emit(std::make_shared<ScalarDiv>(lhs, rhs));
  }
  return GenVInt();
}

void ScalarFloorDiv::UpdateMathInfo() {
  if (!need_eval()) {
    return;
  }
  auto input1 = input_as_sptr<IntSymbol>(0);
  auto input2 = input_as_sptr<IntSymbol>(1);
  auto out = output_as<IntSymbol>();
  auto r = UpdateRange(input1->range_min(), input2->range_min(), input1->range_max(), input2->range_max());
  out->SetRange(r.first, r.second);
}

SymbolPtr ScalarCeilDiv::Eval() {
  // only eval on Building
  auto lhs = input_as_sptr<IntSymbol>(0);
  auto rhs = input_as_sptr<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(CeilDiv(lhs->value(), rhs->value()));
  }
  if (lhs->is_divisible_by(rhs)) {
    DoNotEvalOnRun();
    return Emit(std::make_shared<ScalarDiv>(lhs, rhs));
  }
  // the CeilDiv has not math info, assume the lhs can be divisible by rhs if the env is set.
  if (common::GetEnv("MS_DEV_USE_SYMBOL_CEIL_DIV") == "off") {
    MS_LOG(WARNING) << "Assume the " << lhs->ToString() << " can be divide by " << rhs->ToString() << ".";
    return Emit(std::make_shared<ScalarDiv>(lhs, rhs));
  }
  return GenVInt();
}

SymbolPtr ScalarRealDiv::Eval() {
  auto lhs = input_as_sptr<IntSymbol>(0);
  auto rhs = input_as_sptr<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return FloatSymbol::Make(DivWithCheck(lhs->value(), rhs->value()));
  }
  if (lhs->HasData() && lhs->value() == 0) {
    return FloatSymbol::Make(0);
  }
  if (rhs->HasData() && rhs->value() == 1) {
    DoNotEvalOnRun();
    return Emit(std::make_shared<ScalarCast<FloatSymbol>>(lhs));
  }
  if (*lhs == *rhs) {
    DoNotEvalOnRun();
    return FloatSymbol::Make(1);
  }
  return FloatSymbol::Make(shared_from_this());
}

REG_SYMBOL_OP_BUILDER("ScalarDiv").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarRealDiv>();
REG_SYMBOL_OP_BUILDER("ScalarFloorDiv").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarFloorDiv>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
