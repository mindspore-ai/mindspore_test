/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "infer/make_list.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MakeList, BaseOperator);
AbstractBasePtr MakeListInnerInfer(const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractList>(input_args);
}

class MakeListInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    return MakeListInnerInfer(input_args)->GetShape();
  }

  TypePtr InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    return MakeListInnerInfer(input_args)->GetType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MakeListInnerInfer(input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(MakeList, prim::kPrimMakeList, MakeListInfer, false);
}  // namespace ops
}  // namespace mindspore
