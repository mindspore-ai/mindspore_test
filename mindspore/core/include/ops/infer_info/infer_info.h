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

#ifndef MINDSPORE_CORE_OPS_INFER_INFO_H_
#define MINDSPORE_CORE_OPS_INFER_INFO_H_

#include <vector>
#include <memory>
#include <string>
#include "mindapi/base/shape_vector.h"
#include "utils/log_adapter.h"
#include "utils/value_utils.h"
#include "abstract/abstract_value.h"

#define RETURN_IF_OPTIONAL_HAS_VALUE(opt) \
  do {                                    \
    if (opt.has_value()) {                \
      return opt.value();                 \
    }                                     \
  } while (0)

namespace mindspore::ops {
class InferInfo;
using InferInfoPtr = std::unique_ptr<InferInfo>;
using InferInfoPtrList = std::vector<InferInfoPtr>;
class MS_CORE_API InferInfo {
 public:
  InferInfo() = default;
  virtual ~InferInfo() = default;

  // Shape
  virtual ShapeVector GetShape() = 0;
  virtual bool IsDynamic() = 0;
  virtual bool IsDynamicRank() = 0;

  // Type
  virtual TypeId GetType() = 0;

  // Value
  template <class T>
  std::optional<T> GetScalarValue() {
    if (MS_UNLIKELY(IsNone())) {
      MS_LOG(EXCEPTION) << "Calling GetScalarValue on a None object, " << BaseDebugInfo();
    }
    if (MS_UNLIKELY(IsSequence())) {
      MS_LOG(EXCEPTION) << "Calling GetScalarValue on a sequence, " << BaseDebugInfo();
    }
    return mindspore::GetScalarValue<T>(GetValuePtr());
  }

  template <class T>
  T GetScalarValueWithCheck() {
    T result;
    if (!mindspore::GetScalarValuePtr<T>(GetValuePtr(), &result)) {
      MS_LOG(EXCEPTION) << "Unable to get scalar value, " << BaseDebugInfo();
    }
    return result;
  }

  template <class T>
  std::optional<ArrayValue<T>> GetArrayValue() {
    if (MS_UNLIKELY(IsNone())) {
      MS_LOG(EXCEPTION) << "Calling GetArrayValue on a None object, " << BaseDebugInfo();
    }
    if (GetAbstractPtr()) {
      return mindspore::GetArrayValue<T>(GetAbstractPtr());
    }
    return mindspore::GetArrayValue<T>(GetValuePtr());
  }

  virtual bool IsNone() = 0;

  // Sequence
  virtual bool IsSequence() = 0;
  virtual bool IsDynamicSequence() = 0;
  virtual std::vector<InferInfoPtr> GetSequenceElements() = 0;
  virtual InferInfoPtr GetDynamicSequenceElement() = 0;

  virtual std::string DebugInfo() = 0;

 protected:
  virtual ValuePtr GetValuePtr() = 0;
  virtual AbstractBasePtr GetAbstractPtr() = 0;
  // Debug
  virtual const std::string &BaseDebugInfo() = 0;
};
}  // namespace mindspore::ops
#endif  //  MINDSPORE_CORE_OPS_INFER_INFO_H_
