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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API InnerMoeTokenUnpermute : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~InnerMoeTokenUnpermute() override = default;
  MS_DECLARE_PARENT(InnerMoeTokenUnpermute, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr InnerMoeTokenUnpermute::Eval() {
  constexpr size_t out_shape_rank = 2;
  auto permuted_tokens_shape = input_as<ListSymbol>(kIndex0);
  auto sorted_indices_shape = input_as<ListSymbol>(kIndex1);
  auto probs_shape = input_as<ListSymbol>(kIndex2);  // probs can be None

  SymbolPtrList output_shape(out_shape_rank);
  bool need_eval = false;
  if (permuted_tokens_shape->HasData()) {
    output_shape[kIndex1] = permuted_tokens_shape->symbols().back();
  } else {
    output_shape[kIndex1] = GenVInt();
    need_eval = true;
  }
  if (!sorted_indices_shape->HasData() || !probs_shape->HasData()) {
    output_shape[kIndex0] = GenVInt();
    return GenList(std::move(output_shape));
  }
  if (!need_eval) {
    DoNotEvalOnRun();
  }
  if (probs_shape->size() != 0) {
    // probs_tensor is not None
    auto topk = probs_shape->as<ListSymbol>()->item(1);
    output_shape[0] = Emit(std::make_shared<ScalarDiv>(sorted_indices_shape->item(0), topk));
  } else {
    output_shape[0] = sorted_indices_shape->item(0);
  }
  return ResultIntList(std::move(output_shape));
}

REG_SYMBOL_OP_BUILDER("InnerMoeTokenUnpermute")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape, DependOn::kNone})
  .SetShapeFuncWith<InnerMoeTokenUnpermute>();

REG_SYMBOL_OP_BUILDER("MoeTokenUnpermuteGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
