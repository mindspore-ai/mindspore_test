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

#include "ops/infer_info/abstract_infer_info_adapter.h"
#include <string>
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore::ops {

using abstract::AbstractSequence;
using abstract::AbstractSequencePtr;

BaseShapePtr AbstractInferInfoAdapter::GetShapePtr() {
  RETURN_IF_OPTIONAL_HAS_VALUE(shape_ptr_);
  shape_ptr_ = abs_->GetShape();
  MS_EXCEPTION_IF_NULL(shape_ptr_.value());
  return shape_ptr_.value();
}

TypePtr AbstractInferInfoAdapter::GetTypePtr() {
  RETURN_IF_OPTIONAL_HAS_VALUE(type_ptr_);
  type_ptr_ = abs_->GetType();
  MS_EXCEPTION_IF_NULL(type_ptr_.value());
  return type_ptr_.value();
}

ShapeVector AbstractInferInfoAdapter::GetShape() {
  if (MS_UNLIKELY(IsSequence())) {
    MS_LOG(EXCEPTION) << "Calling GetShape() on a sequence, " << BaseDebugInfo();
  }
  if (MS_UNLIKELY(IsNone())) {
    MS_LOG(EXCEPTION) << "Calling GetShape() on a None object, " << BaseDebugInfo();
  }
  auto shape_ptr = GetShapePtr();
  if (MS_LIKELY(shape_ptr->isa<abstract::TensorShape>())) {
    return shape_ptr->GetShapeVector();
  } else if (shape_ptr->isa<abstract::NoShape>()) {
    return {};
  } else {
    MS_LOG(EXCEPTION) << "Try to get shape from unsupported type: " << shape_ptr->ToString() << ". " << BaseDebugInfo();
  }
}

bool AbstractInferInfoAdapter::IsDynamic() { return GetShapePtr()->IsDynamic(); }

bool AbstractInferInfoAdapter::IsDynamicRank() { return GetShapePtr()->IsDimUnknown(); }

TypeId AbstractInferInfoAdapter::GetType() {
  if (MS_UNLIKELY(IsSequence())) {
    MS_LOG(EXCEPTION) << "Calling GetType() on a sequence, " << BaseDebugInfo();
  }
  if (MS_UNLIKELY(IsNone())) {
    MS_LOG(EXCEPTION) << "Calling GetType() on a None object, " << BaseDebugInfo();
  }
  const auto &type_ptr = GetTypePtr();
  if (type_ptr->isa<TensorType>()) {
    auto tensor_type = type_ptr->cast<TensorTypePtr>();
    return tensor_type->element()->type_id();
  } else if (type_ptr->isa<Number>()) {
    return type_ptr->type_id();
  } else {
    MS_LOG(EXCEPTION) << "Calling GetType() on unsupported value type '" << type_ptr->type_name() << "',"
                      << BaseDebugInfo();
  }
}

bool AbstractInferInfoAdapter::IsNone() {
  RETURN_IF_OPTIONAL_HAS_VALUE(is_none_);
  is_none_ = GetTypePtr()->isa<TypeNone>();
  return is_none_.value();
}

bool AbstractInferInfoAdapter::IsSequence() {
  RETURN_IF_OPTIONAL_HAS_VALUE(is_sequence_);
  is_sequence_ = (GetTypePtr()->object_type() == kObjectTypeTuple) || (GetTypePtr()->object_type() == kObjectTypeList);
  return is_sequence_.value();
}

bool AbstractInferInfoAdapter::IsDynamicSequence() {
  RETURN_IF_OPTIONAL_HAS_VALUE(is_dynamic_seq_);
  if (!IsSequence()) {
    MS_LOG(EXCEPTION) << "Calling IsDynamicSequence() on a non-sequence, " << BaseDebugInfo();
  }
  is_dynamic_seq_ = abs_->GetShape()->isa<abstract::DynamicSequenceShape>();
  return is_dynamic_seq_.value();
}

InferInfoPtrList AbstractInferInfoAdapter::GetSequenceElements() {
  if (!IsSequence()) {
    MS_LOG(EXCEPTION) << "Calling GetSequenceElements() on a non-sequence, " << BaseDebugInfo();
  }
  if (IsDynamicSequence()) {
    MS_LOG(EXCEPTION) << "Sequence is dynamic, unable to get elements, " << BaseDebugInfo();
  }
  InferInfoPtrList elem_infer_infos;
  if (abs_->isa<AbstractSequence>()) {
    auto abstract_sequence = abs_->cast<abstract::AbstractSequencePtr>();
    const auto &elements = abstract_sequence->elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      elem_infer_infos.push_back(std::make_unique<AbstractInferInfoAdapter>(elements[i], op_type_, arg_name_));
    }
  } else {  // KernelTensor
    auto sequence_shape_ptr = abs_->GetShape()->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape_ptr);
    const auto &shapes = sequence_shape_ptr->shape();
    const auto &type_ptr = GetTypePtr();
    TypePtrList types;
    if (type_ptr->isa<Tuple>()) {
      types = type_ptr->cast<TuplePtr>()->elements();
    } else if (type_ptr->isa<List>()) {
      types = type_ptr->cast<ListPtr>()->elements();
    } else {
      MS_LOG(EXCEPTION) << "Failed to get types of list elements from type " << type_ptr->ToString() << ", "
                        << BaseDebugInfo();
    }
    if (shapes.size() != types.size()) {
      MS_LOG(EXCEPTION) << "Shapes number and types number must be equal when calling GetSequenceElements(), but got: "
                        << shapes.size() << " vs " << types.size();
    }
    for (size_t i = 0; i < shapes.size(); ++i) {
      const auto &element = abstract::MakeAbstract(shapes[i], types[i]);
      elem_infer_infos.push_back(
        std::make_unique<AbstractInferInfoAdapter>(element, op_type_, arg_name_ + "_" + std::to_string(i)));
    }
  }
  return elem_infer_infos;
}

InferInfoPtr AbstractInferInfoAdapter::GetDynamicSequenceElement() {
  if (!IsDynamicSequence()) {
    MS_LOG(EXCEPTION) << "Calling GetDynamicSequenceElement on a non-dynamic-sequence, " << BaseDebugInfo();
  }
  auto abstract_sequence = abs_->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abstract_sequence);
  auto element_abs = abstract_sequence->dynamic_len_element_abs();
  return std::make_unique<AbstractInferInfoAdapter>(element_abs, op_type_, arg_name_);
}

ValuePtr AbstractInferInfoAdapter::GetValuePtr() {
  RETURN_IF_OPTIONAL_HAS_VALUE(value_);
  value_ = abs_->GetValue();
  MS_EXCEPTION_IF_NULL(value_);
  return value_.value();
}

AbstractBasePtr AbstractInferInfoAdapter::GetAbstractPtr() { return abs_; }

const std::string &AbstractInferInfoAdapter::BaseDebugInfo() { return base_debug_info_; }

std::string AbstractInferInfoAdapter::DebugInfo() { return BaseDebugInfo() + " -> " + GetAbstractPtr()->ToString(); }
}  // namespace mindspore::ops
