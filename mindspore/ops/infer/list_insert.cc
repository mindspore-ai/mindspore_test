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

#include "infer/list_insert.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
int64_t GetListInsertIndexValue(const ValuePtr &index_value, size_t elements_len, const std::string &prim_name) {
  auto index_opt = GetScalarValue<int64_t>(index_value);
  if (!index_opt.has_value()) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the index value should not be none.";
  }
  auto index = index_opt.value();
  if (index < -static_cast<int64_t>(elements_len)) {
    index = 0;
  }
  if (index > static_cast<int64_t>(elements_len)) {
    index = static_cast<int64_t>(elements_len);
  }
  index = index < 0 ? index + static_cast<int64_t>(elements_len) : index;
  return index;
}

AbstractBasePtr ListInsertInnerInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_len = 3;
  constexpr size_t data_index = 0;
  constexpr size_t index_index = 1;
  constexpr size_t target_index = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto data_abs = dyn_cast<abstract::AbstractSequence>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  auto index_abs = abstract::CheckArg<abstract::AbstractScalar>(prim_name, input_args, index_index);
  auto target_abs = input_args[target_index];
  if (!data_abs->isa<abstract::AbstractSequence>() ||
      (!target_abs->isa<abstract::AbstractScalar>() && !target_abs->isa<abstract::AbstractTensor>())) {
    MS_EXCEPTION(TypeError)
      << "The prim '" << prim_name
      << "', the input_data must be list, index must be scalar, target must be scalar or tensor, but got "
      << data_abs->ToString() << " target is " << target_abs->ToString();
  }

  if (data_abs->dynamic_len()) {
    auto data_element_abs = data_abs->dynamic_len_element_abs();
    if (data_element_abs == nullptr) {
      // The element type of the dynamic length sequence is not determined before list append.
      // Fix the element abstract as the target element
      auto ret = data_abs->Clone();
      ret->cast<abstract::AbstractListPtr>()->set_dynamic_len_element_abs(target_abs);
      return ret;
    }
    // If abstract of element is fixed, the abstract of target should have the same shape and type with the
    // abstract of element.
    CheckAndConvertUtils::CheckAbstractTypeAndShapeSame({data_element_abs, target_abs},
                                                        "For " + prim::kPrimListInsert->ToString(),
                                                        "mutable list existing item", "new added item");
    return data_abs->Clone();
  }

  const auto &elements = data_abs->elements();
  abstract::AbstractBasePtrList abs;
  if (elements.empty()) {
    abs.push_back(target_abs);
    return std::make_shared<abstract::AbstractList>(abs);
  }

  auto first_element = elements[0];
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(
    {first_element, target_abs}, "For " + prim::kPrimListInsert->ToString(), "list existing item", "new added item");
  for (size_t i = 0; i < data_abs->size(); ++i) {
    abs.push_back(data_abs->elements()[i]);
  }
  ValuePtr index_value = index_abs->GetValue();
  if (index_value->ContainsValueAny()) {
    abs.push_back(target_abs);
    auto new_abs = std::make_shared<abstract::AbstractList>(abs);
    return CheckAndConvertUtils::BroadenAllSequenceElements(new_abs);
  }
  int64_t index = GetListInsertIndexValue(index_value, elements.size(), prim_name);
  (void)abs.insert(abs.begin() + index, target_abs);
  return std::make_shared<abstract::AbstractList>(abs);
}

class ListInsertInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_shape = input_args[kIndex0]->GetShape();
    auto target_shape = input_args[kIndex2]->GetShape();
    auto index_value = input_args[kIndex1]->GetValue();
    auto list_shape = input_shape->cast<abstract::SequenceShapePtr>()->shape();
    auto index = GetListInsertIndexValue(index_value, list_shape.size(), primitive->name());
    list_shape.insert(list_shape.begin() + index, target_shape);
    return std::make_shared<abstract::ListShape>(list_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_type = input_args[kIndex0]->GetType();
    auto target_type = input_args[kIndex2]->GetType();
    auto index_value = input_args[kIndex1]->GetValue();
    auto list_type = input_type->cast<ListPtr>()->elements();
    auto index = GetListInsertIndexValue(index_value, list_type.size(), primitive->name());
    list_type.insert(list_type.begin() + index, target_type);
    return std::make_shared<List>(list_type);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ListInsertInnerInfer(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(ListInsert, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ListInsert, prim::kPrimListInsert, ListInsertInfer, false);
}  // namespace ops
}  // namespace mindspore
