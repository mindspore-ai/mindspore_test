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

#include "infer/send.h"

#include <set>
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Send, BaseOperator);
class SendInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckArgsType(prim_name, input_args, kIndex0, kObjectTypeTensorType);
    auto x = input_args[0]->GetShape();
    MS_EXCEPTION_IF_NULL(x);
    auto shape_element = x->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_element);

    // Add shape attr for kbk mode.
    if (primitive->GetAttr("shape") == nullptr) {
      auto shape = MakeValue(x->GetShapeVector());
      MS_EXCEPTION_IF_NULL(shape);
      primitive->set_attr("shape", shape);
    }
    return shape_element;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->GetType();
    const std::set<TypePtr> valid_types = {kInt8, kInt32, kFloat16, kFloat32, kFloat64, kBFloat16};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
    // Add dtype attr for kbk mode.
    if (prim->GetAttr("dtype") == nullptr) {
      MS_EXCEPTION_IF_NULL(x_type);
      prim->set_attr("dtype", x_type);
    }
    return x_type->Clone();
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Send, prim::kPrimSend, SendInfer, false);

}  // namespace ops
}  // namespace mindspore
