/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Gather : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Gather(const SymbolPtr &param, const SymbolPtr &indices, const SymbolPtr &axis, const SymbolPtr &batch_dims)
      : InferShapeOp({param, indices, axis, batch_dims}) {}
  ~Gather() override = default;
  MS_DECLARE_PARENT(Gather, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Gather::Eval() {
  auto params = input_as<ListSymbol>(kIndex0);
  auto indices = input_as<ListSymbol>(kIndex1);
  auto axis = input_as<IntSymbol>(kIndex2);
  auto batch_dims = input_as<IntSymbol>(kIndex3)->value();
  if (!params->HasData() || !indices->HasData() || !axis->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto axis_val = LongToSize(NormAxis(axis->value(), params->size()));
  batch_dims = NormAxis(batch_dims, indices->size());
  SymbolPtrList result;
  result.reserve(params->size() + indices->size());
  MS_EXCEPTION_IF_CHECK_FAIL(axis_val < params->size(), "axis out of params size.");
  for (size_t i = 0; i < axis_val; i++) {
    (void)result.emplace_back(params->symbols()[i]);
  }
  for (size_t i = LongToSize(batch_dims); i < indices->size(); i++) {
    (void)result.emplace_back(indices->symbols()[i]);
  }
  for (size_t i = axis_val + 1; i < params->size(); i++) {
    (void)result.emplace_back(params->symbols()[i]);
  }
  return ResultIntList(std::move(result));
}

class OPS_API GatherNd : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  GatherNd(const SymbolPtr &x, const SymbolPtr &indices) : InferShapeOp({x, indices}) {}
  ~GatherNd() override = default;
  MS_DECLARE_PARENT(GatherNd, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr GatherNd::Eval() {
  auto input_x_shape = input_as<ListSymbol>(kIndex0);
  auto indices_shape = input_as<ListSymbol>(kIndex1);
  if (!input_x_shape->HasData() || !indices_shape->HasData()) {
    return GenVList();
  }
  size_t indices_rank = indices_shape->size();
  auto indices_end_value = indices_rank > 0 ? indices_shape->symbols().back()->as_sptr<IntSymbol>() : kSym1;
  if (!indices_end_value->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  SymbolPtrList output_shape;
  for (size_t i = 0; i + 1 < indices_rank; i++) {
    (void)output_shape.emplace_back(indices_shape->symbols()[i]);
  }
  for (size_t i = LongToSize(indices_end_value->value()); i < input_x_shape->size(); i++) {
    (void)output_shape.emplace_back(input_x_shape->symbols()[i]);
  }
  return ResultIntList(std::move(output_shape));
}

REG_SYMBOL_OP_BUILDER("Gather")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto params = b->GetInputShape(kIndex0);
    auto indices = b->GetInputShape(kIndex1);
    auto axis = b->GetInputValue(kIndex2);
    auto batch_dims = b->GetInputOrAttr(kIndex3, kAttrBatchDims);
    MS_EXCEPTION_IF_NULL(batch_dims);
    return b->Emit(std::make_shared<Gather>(params, indices, axis, batch_dims));
  });

REG_SYMBOL_OP_BUILDER("GatherD")
  .SetShapeDepend({DependOn::kNone, DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);

REG_SYMBOL_OP_BUILDER("GatherNd").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<GatherNd>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
