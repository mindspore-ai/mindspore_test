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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_add.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API RepeatInterleave : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  RepeatInterleave(const SymbolPtr &x, const SymbolPtr &repeat, const SymbolPtr &dim)
      : InferShapeOp({x, repeat, dim}) {}
  ~RepeatInterleave() override = default;
  MS_DECLARE_PARENT(RepeatInterleave, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr RepeatInterleave::Eval() {
  auto x = input_as_sptr<ListSymbol>(kIndex0);
  auto repeats = input_as_sptr<ListSymbol>(kIndex1);
  auto dim = input(kIndex2);
  if (!x->HasData()) {
    return GenVList();
  }
  if (!repeats->HasData()) {
    return dim->is<NoneSymbol>() ? ListSymbol::Make({GenVInt()}) : GenVIntList(x->size());
  }
  if (!dim->is<NoneSymbol>() && !dim->HasData()) {
    return GenVIntList(x->size());
  }

  DoNotEvalOnRun();
  auto repeat_sum = Accumulate<ScalarAdd>(repeats->symbols(), emitter());
  auto repeats_numel = repeats->size();
  if (dim->is<NoneSymbol>()) {
    if (repeats_numel == 1) {
      auto x_numel = Accumulate<ScalarMul>(x->symbols(), emitter());
      return GenList({Emit(std::make_shared<ScalarMul>(x_numel, repeat_sum))});
    } else {
      return GenList({repeat_sum});
    }
  } else {
    auto result = x->symbols();
    auto axis = LongToSize(NormAxis(AsInt(dim), x->size()));
    if (repeats_numel == 1) {
      result[axis] = Emit(std::make_shared<ScalarMul>(result[axis], repeat_sum));
    } else {
      result[axis] = repeat_sum;
    }
    return ResultIntList(std::move(result));
  }
}

REG_SYMBOL_OP_BUILDER("RepeatInterleaveInt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) {
    auto x = b->GetInputShape(kIndex0);
    auto repeat = b->GetInputValue(kIndex1);
    auto dim = b->GetInputValue(kIndex2);
    return b->Emit(std::make_shared<RepeatInterleave>(x, ListSymbol::Make({repeat}), dim));
  });

REG_SYMBOL_OP_BUILDER("RepeatInterleaveTensor")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<RepeatInterleave>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
