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

#include <map>
#include <string>
#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "mindspore/ccsrc/utils/symbol_engine/symbol_engine_impl_pi.h"

namespace mindspore::symshape::test {
class TestSymbolEnginePI : public UT::Common {};
/// Feature: Test symbol engine for pijit
/// Description: add scalareq as guard, then check guard with real inputs.
/// Expectation: infer guard success.
TEST_F(TestSymbolEnginePI, concat) {
  mindspore::test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {1, 32});
  auto x0abs = x0->abstract();
  auto x0hint_abs = x0abs->Clone();
  auto x0shape = x0abs->GetShape();
  x0shape->Broaden();
  x0abs->set_shape(x0shape);
  auto x1abs = x1->abstract();
  auto x1hint_abs = x1abs->Clone();
  auto x1shape = x1abs->GetShape();
  x1shape->Broaden();
  x1abs->set_shape(x1shape);
  auto v1 = c.NewValueNode(MakeValue<int64_t>(1));
  auto op = c.NewCNodeWithBuildInfo("Concat", {x0, x1, v1}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Shape", {op}, {});
  auto op2 = c.NewCNodeWithBuildInfo("TupleGetItem", {op1, v1}, {});
  auto v4 = c.NewValueNode(MakeValue<int64_t>(64));
  auto guard = c.NewCNodeWithBuildInfo("ScalarEq", {op2, v4}, {});
  c.SetOutput(guard);
  auto fg = c.GetGraph();
  auto engine = symshape::SymbolEnginePIJIT::Build(fg);
  AbstractBasePtrList inputs_abs;
  inputs_abs.emplace_back(x0hint_abs);
  inputs_abs.emplace_back(x1hint_abs);
  engine->AddInputAbs(x0abs, x0hint_abs);
  engine->AddInputAbs(x1abs, x1hint_abs);
  engine->BuildCNodeSymbol(op);
  engine->BuildCNodeSymbol(op1);
  engine->BuildCNodeSymbol(op2);
  engine->BuildCNodeSymbol(guard);
  auto sym_cond = guard->abstract()->GetSymbolicValue()->as_sptr<BoolSymbol>();
  ASSERT_FALSE(sym_cond->HasData());
  ASSERT_TRUE(engine->SupportInfer());
  ASSERT_TRUE(engine->CheckCondition(inputs_abs, sym_cond));
  auto x2 = c.NewTensorInput("x0", kFloat32, {1, 3});
  auto x3 = c.NewTensorInput("x1", kFloat32, {1, 3});
  AbstractBasePtrList inputs_abs2;
  inputs_abs2.emplace_back(x2->abstract());
  inputs_abs2.emplace_back(x3->abstract());
  ASSERT_FALSE(engine->CheckCondition(inputs_abs2, sym_cond));
  ASSERT_TRUE(engine->CheckCondition(inputs_abs, sym_cond));
}
}  // namespace mindspore::symshape::test
