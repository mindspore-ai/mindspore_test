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

#ifndef MINDSPORE_CORE_OPS_ABSTRACT_INFER_INFO_ADAPTER_H_
#define MINDSPORE_CORE_OPS_ABSTRACT_INFER_INFO_ADAPTER_H_

#include <vector>
#include <memory>
#include <string>
#include "ops/infer_info/infer_info.h"
#include "mindapi/base/shape_vector.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
class AbstractInferInfoAdapter;
using AbstractInferInfoAdapterPtr = std::shared_ptr<AbstractInferInfoAdapter>;
class MS_CORE_API AbstractInferInfoAdapter : public InferInfo {
 public:
  AbstractInferInfoAdapter() = delete;
  AbstractInferInfoAdapter(const AbstractBasePtr &abs, const std::string &op_type, const std::string &arg_name)
      : abs_(abs), op_type_(op_type), arg_name_(arg_name) {
    base_debug_info_ = "op type: [" + op_type_ + "], arg name: [" + arg_name_ + "]";
  }

  // Shape
  ShapeVector GetShape() override;
  bool IsDynamic() override;
  bool IsDynamicRank() override;

  // Type
  TypeId GetType() override;

  // Value
  bool IsNone() override;

  // Sequence
  bool IsSequence() override;
  bool IsDynamicSequence() override;
  std::vector<InferInfoPtr> GetSequenceElements() override;
  InferInfoPtr GetDynamicSequenceElement() override;

  std::string DebugInfo() override;

 private:
  const std::string &BaseDebugInfo() override;
  ValuePtr GetValuePtr() override;
  AbstractBasePtr GetAbstractPtr() override;
  BaseShapePtr GetShapePtr();
  TypePtr GetTypePtr();

  const AbstractBasePtr abs_;
  const std::string op_type_;
  const std::string arg_name_;
  std::string base_debug_info_;

  std::optional<BaseShapePtr> shape_ptr_;
  std::optional<TypePtr> type_ptr_;
  std::optional<bool> is_dynamic_seq_;
  std::optional<bool> is_none_;
  std::optional<bool> is_sequence_;
  std::optional<ValuePtr> value_;
};
}  // namespace mindspore::ops
#endif  //  MINDSPORE_CORE_OPS_ABSTRACT_INFER_INFO_ADAPTER_H_
