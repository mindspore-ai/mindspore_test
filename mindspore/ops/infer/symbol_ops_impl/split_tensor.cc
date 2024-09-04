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

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API SplitTensor : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~SplitTensor() override = default;
  MS_DECLARE_PARENT(SplitTensor, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr SplitTensor::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto split_size = input_as<IntSymbol>(kIndex1)->value();
  if (!x->HasData()) {
    return GenVList();
  }
  auto axis = LongToSize(NormAxis(input_as<IntSymbol>(kIndex2)->value(), x->size()));
  auto x_pos = x->item_as<IntSymbol>(axis)->value();
  DoNotEvalOnRun();
  auto out = x->symbols();
  out[axis] = input(kIndex1);
  if (x_pos % split_size == 0) {
    return GenList(SymbolPtrList(x_pos / split_size, GenList(std::move(out))));
  } else {
    SymbolPtrList result(x_pos / split_size, GenList(out));
    out[axis] = GenInt(x_pos % split_size);
    result.push_back(GenList(std::move(out)));
    return GenList(std::move(result));
  }
}

REG_SYMBOL_OP_BUILDER("SplitTensor")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<SplitTensor>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
