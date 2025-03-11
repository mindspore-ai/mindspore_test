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
  explicit Flatten(const SymbolPtr &input) : InferShapeOp({input}) {}
  Flatten(const SymbolPtr &input, const SymbolPtr &start, const SymbolPtr &end) : InferShapeOp({input, start, end}) {}
  ~Flatten() override = default;
  MS_DECLARE_PARENT(Flatten, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Flatten::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  bool is_flatten_ext = input_num() > 1;
  auto start_sym = is_flatten_ext ? input_as_sptr<IntSymbol>(kIndex1) : kSym1;
  auto end_sym = is_flatten_ext ? input_as_sptr<IntSymbol>(kIndex2) : kSymNeg1;
  if (!x->HasData()) {
    return GenVList();
  }
  if (x->size() == 0) {
    return GenList({kSym1});
  }
  if (x->size() == 1) {
    DoNotEvalOnRun();
    if (is_flatten_ext) {
      return input(kIndex0);
    } else {
      // the Flatten's output is always 2-D
      return GenList(SymbolPtrList{x->item(0), kSym1});
    }
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
  result.reserve(x->size() - end + start);
  for (size_t i = 0; i < x->size(); i++) {
    // the flatten range is [start, end].
    // when i == start, put the x[i] into result. when i > start, we can use result.back() = result.back() * x[i]
    if (start < i && i <= end) {
      result.back() = Emit(std::make_shared<ScalarMul>(result.back(), x->item(i)));
    } else {
      (void)result.emplace_back(x->item(i));
    }
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Flatten").SetShapeDepend({DependOn::kShape}).SetShapeFuncWith<Flatten>();
REG_SYMBOL_OP_BUILDER("FlattenExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Flatten>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
