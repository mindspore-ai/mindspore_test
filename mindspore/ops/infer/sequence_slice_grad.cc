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

#include "infer/sequence_slice_grad.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr SliceGradInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 5;
  constexpr size_t x_index = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  return input_args[x_index]->Clone();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceSliceGrad, BaseOperator);
class SequenceSliceGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_num = 5;
    constexpr size_t x_index = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    return input_args[x_index]->GetShape()->Clone();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_num = 5;
    constexpr size_t x_index = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    return input_args[x_index]->GetType()->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceGradInferInner(primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {2, 3, 4}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceSliceGrad, prim::kPrimSequenceSliceGrad, SequenceSliceGradInfer, false);
}  // namespace ops
}  // namespace mindspore
