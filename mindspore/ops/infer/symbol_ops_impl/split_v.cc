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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_add.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_sub.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API SplitV : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  SplitV(const SymbolPtr &x, const SymbolPtr &axis, const SymbolPtr &size_splits)
      : InferShapeOp({x, axis, size_splits}) {}
  ~SplitV() override = default;
  MS_DECLARE_PARENT(SplitV, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr SplitV::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto size_splits = input_as<ListSymbol>(kIndex2)->symbols();
  auto out_num = size_splits.size();
  if (!x->HasData()) {
    SymbolPtrList ret(out_num);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = GenVList();
    }
    return GenList(std::move(ret));
  }
  auto axis = LongToSize(NormAxis(input_as<IntSymbol>(kIndex1)->value(), x->size()));
  DoNotEvalOnRun();
  MS_EXCEPTION_IF_CHECK_FAIL(size_splits.size() == out_num, "The size_splits.size() does not equals to num_split.");
  auto iter = std::find_if(size_splits.begin(), size_splits.end(),
                           [](const SymbolPtr &s) { return s->as<IntSymbol>()->is_negative(); });
  auto unknown_idx = iter - size_splits.begin();
  if (iter != size_splits.end()) {
    // put a "1" into size_splits to accumulate with the inner "-1".
    size_splits.push_back(kSym1);
    auto sizes = Accumulate<ScalarAdd>(size_splits, emitter());
    size_splits[unknown_idx] = Emit(std::make_shared<ScalarSub>(x->item(axis), sizes));
  }
  SymbolPtrList result(out_num);
  for (size_t i = 0; i < out_num; i++) {
    auto syms = x->symbols();
    syms[axis] = size_splits[i];
    result[i] = GenList(std::move(syms));
  }
  return GenList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("SplitV").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto x = b->GetInputShape(kIndex0);
  auto split_dim = b->GetAttr("split_dim");
  MS_EXCEPTION_IF_NULL(split_dim);
  auto size_splits = b->GetAttr("size_splits");
  MS_EXCEPTION_IF_NULL(size_splits);
  return b->Emit(std::make_shared<SplitV>(x, split_dim, size_splits));
});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
