/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Flatten : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Flatten(const SymbolPtr &input, const SymbolPtr &start, const SymbolPtr &end) : InferShapeOp({input, start, end}) {}
  ~Flatten() override = default;
  MS_DECLARE_PARENT(Flatten, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Flatten::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto start_sym = input_as<IntSymbol>(kIndex1);
  auto end_sym = input_as<IntSymbol>(kIndex2);
  if (!x->HasData()) {
    return GenVList();
  }
  if (x->size() == 0) {
    return GenList({kSym1});
  }
  if (x->size() == 1) {
    DoNotEvalOnRun();
    return input(kIndex0);
  }
  if (!start_sym->HasData() || !end_sym->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto start = LongToSize(NormAxis(start_sym->value(), x->size()));
  auto end = LongToSize(NormAxis(end_sym->value(), x->size()));
  if (start > end) {
    MS_EXCEPTION(ValueError) << "For 'Flatten', 'start_dim' cannot come after 'end_dim', got " << start << " vs "
                             << end;
  }
  if (start == end) {
    return input(kIndex0);
  }
  SymbolPtrList result;
  result.reserve(x->size());
  for (size_t i = 0; i < x->size(); i++) {
    if (start < i && i < end) {
      result.back() = Emit(std::make_shared<ScalarMul>(result.back(), x->item(i)));
    } else {
      (void)result.emplace_back(x->item(i));
    }
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Flatten").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  return b->Emit(std::make_shared<Flatten>(b->GetInputShape(kIndex0), kSym1, kSymNeg1));
});
REG_SYMBOL_OP_BUILDER("FlattenExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Flatten>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
