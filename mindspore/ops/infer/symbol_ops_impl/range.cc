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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_sub.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_min.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Range : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Range(const SymbolPtr &start, const SymbolPtr &end, const SymbolPtr &step) : InferShapeOp({start, end, step}) {}
  ~Range() override = default;
  MS_DECLARE_PARENT(Range, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Range::Eval() {
  auto start = input(kIndex0);
  auto end = input(kIndex1);
  auto step = input(kIndex2);
  DoNotEvalOnRun();
  // range length = (end - start) / step.  (to ceil)
  auto len = Emit(std::make_shared<ScalarSub>(end, start));
  len = Emit(std::make_shared<ScalarCeilDiv>(len, step));
  if (input_num() > kIndex3) {
    len = Emit(std::make_shared<ScalarMin>(len, input(kIndex3)));
  }
  return GenList({len});
}

REG_SYMBOL_OP_BUILDER("Range").SetShapeDependN<DependOn::kValue, 4>().SetShapeFuncWith<Range>();

SymbolPtr ArangeInferShape(OperationBuilder *b) {
  auto start = b->GetInputValue(kIndex0);
  auto end = b->GetInputValue(kIndex1);
  auto step = b->GetInputValue(kIndex2);
  if (start->is<IntSymbol>()) {
    return b->Emit(std::make_shared<Range>(start, end, step));
  }
  return nullptr;
}
REG_SYMBOL_OP_BUILDER("Arange").SetShapeDependN<DependOn::kValue, 3>().SetShapeFunc(ArangeInferShape);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
