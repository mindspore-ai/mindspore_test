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
#include "mindspore/ops/infer/symbol_ops_impl/scalar_min.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API MoeTokenPermute : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~MoeTokenPermute() override = default;
  MS_DECLARE_PARENT(MoeTokenPermute, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr MoeTokenPermute::Eval() {
  constexpr size_t out_permuted_tokens_shape_rank = 2;
  constexpr size_t out_sorted_indices_shape_rank = 1;
  auto tokens_shape = input_as<ListSymbol>(kIndex0);
  auto indices_shape = input_as<ListSymbol>(kIndex1);
  auto num_out_tokens = input(kIndex2);
  SymbolPtrList out_permuted_tokens_shape(out_permuted_tokens_shape_rank);
  bool need_eval = false;
  if (tokens_shape->HasData()) {
    out_permuted_tokens_shape[kIndex1] = tokens_shape->item(kIndex1);
  } else {
    out_permuted_tokens_shape[kIndex1] = GenVInt();
    need_eval = true;
  }
  if (!indices_shape->HasData()) {
    out_permuted_tokens_shape[kIndex0] = GenVInt();
    return GenList({GenList(std::move(out_permuted_tokens_shape)), GenVIntList(out_sorted_indices_shape_rank)});
  }
  if (!need_eval) {
    DoNotEvalOnRun();
  }
  SymbolPtr topk = indices_shape->size() > 1 ? indices_shape->item(kIndex1) : kSym1;
  auto total_length = Emit(std::make_shared<ScalarMul>(topk, indices_shape->item(0)));
  if (num_out_tokens->is<NoneSymbol>()) {
    num_out_tokens = total_length;
  } else {
    num_out_tokens = Emit(std::make_shared<ScalarMin>(num_out_tokens, total_length));
  }
  out_permuted_tokens_shape[kIndex0] = num_out_tokens;
  return GenList({GenList(std::move(out_permuted_tokens_shape)), GenList({total_length})});
}

REG_SYMBOL_OP_BUILDER("MoeTokenPermute")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kNone})
  .SetShapeFuncWith<MoeTokenPermute>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
