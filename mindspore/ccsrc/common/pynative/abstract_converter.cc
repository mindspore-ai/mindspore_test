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

#include "include/common/pynative/abstract_converter.h"
#include <vector>
#include "abstract/abstract_value.h"

namespace mindspore {
namespace pynative {
void AbstractConverter::CacheAbstract(const AbstractBasePtr &abstract) {}

AbstractBasePtr AbstractConverter::ConvertAbstract(const ValuePtr &t) {
  MS_EXCEPTION_IF_NULL(t);
  if (t->isa<BaseTensor>()) {
    auto tensor = t->cast<BaseTensorPtr>();
    return ConvertAbstract(tensor);
  }
  if (t->isa<ValueTuple>()) {
    auto tuple = t->cast<ValueTuplePtr>();
    return ConvertAbstract(tuple);
  }
  return t->ToAbstract();
}

// Tensor is held by Abstract, may lead to memory leak.
AbstractBasePtr AbstractConverter::ConvertAbstract(const BaseTensorPtr &t) {
  MS_EXCEPTION_IF_NULL(t);
  auto abs = t->ToAbstract();
  abs->set_value(kValueAny);
  return abs;
}

AbstractBasePtr AbstractConverter::ConvertAbstract(const tensor::TensorPtr &t) {
  MS_EXCEPTION_IF_NULL(t);
  auto abs = t->ToAbstract();
  abs->set_value(kValueAny);
  return abs;
}

AbstractBasePtr AbstractConverter::ConvertAbstract(const ValueTuplePtr &t) {
  MS_EXCEPTION_IF_NULL(t);
  AbstractBasePtrList abs_list(t->value().size());
  for (size_t i = 0; i < t->value().size(); ++i) {
    auto &val = t->value()[i];
    auto abs = val->ToAbstract();
    if (val->isa<tensor::BaseTensor>()) {
      abs->set_value(kValueAny);
    }
    abs_list[i] = abs;
  }
  return std::make_shared<abstract::AbstractTuple>(abs_list);
}
}  // namespace pynative
}  // namespace mindspore
