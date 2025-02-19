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

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Pad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Pad(const SymbolPtr &input, const SymbolPtr &padding) : InferShapeOp({input, padding}) {}
  ~Pad() override = default;
  MS_DECLARE_PARENT(Pad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Pad::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  auto paddings = input_as<ListSymbol>(kIndex1);
  if (!input->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto result = input->symbols();
  for (size_t i = 0; i < input->size(); i++) {
    auto padding = paddings->item_as<ListSymbol>(i);
    auto p = Emit(std::make_shared<ScalarAdd>(padding->item(0), padding->item(1)));
    result[i] = Emit(std::make_shared<ScalarAdd>(result[i], p));
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Pad").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto input = b->GetInputShape(kIndex0);
  auto paddings = b->GetAttr("paddings");
  MS_EXCEPTION_IF_NULL(paddings);
  return b->Emit(std::make_shared<Pad>(input, paddings));
});

class OPS_API CommonPadOp : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  // contiguous is used in op PadV3.
  CommonPadOp(const SymbolPtr &input, const SymbolPtr &padding, const SymbolPtr &contiguous)
      : InferShapeOp({input, padding, contiguous}) {}
  ~CommonPadOp() override = default;
  MS_DECLARE_PARENT(CommonPadOp, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr CommonPadOp::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  auto padding = input_as<ListSymbol>(kIndex1);
  auto contiguous = input_as<BoolSymbol>(kIndex2)->value();
  const size_t kNum2 = 2;
  if (!input->HasData()) {
    return GenVList();
  }
  if (!padding->HasData()) {
    return GenVIntList(input->size());
  }
  if (padding->size() % kNum2 != 0 || padding->size() > input->size() * kNum2) {
    MS_LOG(INTERNAL_EXCEPTION) << "For Pad op, the padding size should be even number and less-equal to "
                               << (input->size() * kNum2) << ", but got " << padding->size();
  }
  DoNotEvalOnRun();

  // when input shape is (A, B, C), contiguous=true
  // paddings: (p0, p1)                 -- pads dim C
  // paddings: (p0, p1, p2, p3)         -- the (p2,p3) pads dim B, the (p0,p1) pads dim C.
  // paddings: (p0, p1, p2, p3, p4, p5) -- the (p4,p5) pads dim A, the (p2,p3) pads dim B, the (p0,p1) pads dim C.
  SymbolPtrList result = input->symbols();
  auto result_iter = result.rbegin();
  for (size_t i = 0; i < input->size(); i++, ++result_iter) {
    size_t begin_i;
    size_t end_i;
    if (contiguous) {
      // the padding is [begin_0, end_0, begin_1, end_1, ..., begin_n, end_n]
      begin_i = i * kNum2;
      end_i = begin_i + 1;
    } else {
      // the padding is [begin_0, begin_1, ..., begin_n, end_0, end_1, ..., end_n]
      begin_i = i;
      end_i = i + padding->size() / kNum2;
    }
    if (end_i >= padding->size()) {
      break;
    }
    auto p = Emit(std::make_shared<ScalarAdd>(padding->symbols()[begin_i], padding->symbols()[end_i]));
    *result_iter = Emit(std::make_shared<ScalarAdd>(*result_iter, p));
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("PadV3")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto input = b->GetInputShape(kIndex0);
    auto padding = b->GetInputValue(kIndex1);
    auto contiguous = b->GetAttr("paddings_contiguous");
    MS_EXCEPTION_IF_NULL(contiguous);
    return b->Emit(std::make_shared<CommonPadOp>(input, padding, contiguous));
  });

SymbolPtr PadOpsShapeBuilder(OperationBuilder *b) {
  auto input = b->GetInputShape(kIndex0);
  auto padding = b->GetInputValue(kIndex1);
  return b->Emit(std::make_shared<CommonPadOp>(input, padding, BoolSymbol::Make(true)));
}

REG_SYMBOL_OP_BUILDER("ConstantPadNd")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReflectionPad1D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReflectionPad2D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReflectionPad3D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReplicationPad1D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReplicationPad2D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
REG_SYMBOL_OP_BUILDER("ReplicationPad3D")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(PadOpsShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
