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
class OPS_API Chunk : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Chunk() override = default;
  MS_DECLARE_PARENT(Chunk, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Chunk::Eval() {
  auto input_shape = input_as<ListSymbol>(kIndex0);
  if (!input_shape->HasData()) {
    return nullptr;
  }
  DoNotEvalOnRun();
  auto chunks = input_as<IntSymbol>(kIndex1)->value();
  auto rank = input_shape->size();
  auto dim = NormAxis(input_as<IntSymbol>(kIndex2)->value(), rank);
  // chunk does not support dynamic dim.
  auto dim_size = input_shape->item_as<IntSymbol>(LongToSize(dim))->value();
  int64_t each_size = (dim_size + chunks - 1) / chunks;
  // the check in "ops_func_impl/chunk.cc" is "each_size == 0 && dim_size == 0", but it will lead to a static-check
  // warning in following code of "DivisionByZero"  when each_size=0.
  // In this op, chunks is always greater than 0, so when each_size is 0, the dim_size is also 0.
  if (each_size == 0) {
    return GenList(SymbolPtrList(LongToSize(chunks), kSym0));
  }
  auto actual_chunks = std::max<int64_t>((dim_size + each_size - 1) / each_size, 1);
  SymbolPtrList each_shape = input_shape->symbols();
  each_shape[dim] = GenInt(each_size);
  SymbolPtrList result(LongToSize(actual_chunks), GenList(std::move(each_shape)));
  int64_t last_split_size = each_size - (each_size * actual_chunks - dim_size);
  auto last_shape = input_shape->symbols();
  last_shape[dim] = GenInt(last_split_size);
  result.back() = GenList(std::move(last_shape));
  return GenList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Chunk")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Chunk>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
