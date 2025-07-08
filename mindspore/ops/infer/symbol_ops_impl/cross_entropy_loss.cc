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
#include "mindspore/core/include/mindapi/base/types.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API CrossEntropyLoss : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  CrossEntropyLoss(const SymbolPtr &x, const SymbolPtr &k) : InferShapeOp({x, k}) {}
  ~CrossEntropyLoss() override = default;
  MS_DECLARE_PARENT(CrossEntropyLoss, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  std::pair<SymbolPtr, SymbolPtr> CalcNC(const ListSymbol *input_shape, const ListSymbol *target_shape,
                                         const ListSymbol *weight_shape);
  bool eval_on_run_{false};
};

std::pair<SymbolPtr, SymbolPtr> CrossEntropyLoss::CalcNC(const ListSymbol *input_shape, const ListSymbol *target_shape,
                                                         const ListSymbol *weight_shape) {
  SymbolPtr N;
  SymbolPtr C;
  if (input_shape->HasData()) {
    N = input_shape->item(kIndex0);
    C = input_shape->item(kIndex1);
    if (!N->HasData() && target_shape->HasData()) {
      auto t = target_shape->item(kIndex0);
      if (t->HasData()) {
        N = t;
      } else {
        N->as<IntSymbol>()->SetEqual(t->as_sptr<IntSymbol>());
      }
    }
    if (!C->HasData() && weight_shape->HasData() && weight_shape->size() > 0) {
      auto t = weight_shape->item(kIndex0);
      if (t->HasData()) {
        C = t;
      } else {
        C->as<IntSymbol>()->SetEqual(t->as_sptr<IntSymbol>());
      }
    }
  } else {
    if (target_shape->HasData()) {
      N = target_shape->item(kIndex0);
    } else {
      N = GenVInt();
      eval_on_run_ = true;
    }
    if (weight_shape->HasData() && weight_shape->size() > 0) {
      C = weight_shape->item(kIndex0);
    } else {
      C = GenVInt();
      eval_on_run_ = true;
    }
  }
  return std::make_pair(N, C);
}

SymbolPtr CrossEntropyLoss::Eval() {
  auto input_shape = input_as<ListSymbol>(kIndex0);
  auto target_shape = input_as<ListSymbol>(kIndex1);
  auto weight_shape = input_as<ListSymbol>(kIndex2);  // weight can be none
  auto reduction_opt = input_as<IntSymbol>(kIndex3);
  auto lse_for_zloss_opt = input_as<FloatSymbol>(kIndex6);
  auto return_zloss_opt = input_as<BoolSymbol>(kIndex7);

  eval_on_run_ = false;
  auto ret = CalcNC(input_shape, target_shape, weight_shape);
  SymbolPtr N = ret.first;
  SymbolPtr C = ret.second;

  SymbolPtr out_loss_shape;
  SymbolPtr out_log_prob_shape = ListSymbol::Make({N, C});
  SymbolPtr out_zloss_shape;
  SymbolPtr out_lse_for_zloss_shape;

  if (!reduction_opt->HasData()) {
    out_loss_shape = GenVIntList(1);
    eval_on_run_ = true;
  } else {
    out_loss_shape = (reduction_opt->value() != static_cast<int64_t>(Reduction::NONE)) ? ListSymbol::Make({kSym1})
                                                                                       : ListSymbol::Make({N});
  }

  if (!return_zloss_opt->HasData()) {
    out_zloss_shape = GenVIntList(1);
    eval_on_run_ = true;
  } else {
    out_zloss_shape = return_zloss_opt->value() ? ListSymbol::Make({N}) : ListSymbol::Make({kSym0});
  }

  if (!lse_for_zloss_opt->HasData()) {
    out_lse_for_zloss_shape = GenVIntList(1);
    eval_on_run_ = true;
  } else {
    out_lse_for_zloss_shape = (lse_for_zloss_opt->value() != 0.f) ? ListSymbol::Make({N}) : ListSymbol::Make({kSym0});
  }

  if (!eval_on_run_) {
    DoNotEvalOnRun();
  }
  return ListSymbol::Make({out_loss_shape, out_log_prob_shape, out_zloss_shape, out_lse_for_zloss_shape});
}

REG_SYMBOL_OP_BUILDER("CrossEntropyLoss")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<CrossEntropyLoss>();
REG_SYMBOL_OP_BUILDER("CrossEntropyLossGrad")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
