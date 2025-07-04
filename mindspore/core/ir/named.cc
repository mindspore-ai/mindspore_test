/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "ir/named.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"

namespace mindspore {
bool Named::operator==(const Value &other) const {
  if (other.isa<Named>()) {
    auto &other_named = static_cast<const Named &>(other);
    return *this == other_named;
  }
  return false;
}

abstract::AbstractBasePtr None::ToAbstract() { return std::make_shared<abstract::AbstractNone>(); }

abstract::AbstractBasePtr Null::ToAbstract() { return std::make_shared<abstract::AbstractNull>(); }

abstract::AbstractBasePtr Ellipsis::ToAbstract() { return std::make_shared<abstract::AbstractEllipsis>(); }

abstract::AbstractBasePtr Functional::ToAbstract() {
  return std::make_shared<abstract::FunctionalAbstractClosure>(name(), is_method());
}

abstract::AbstractBasePtr MindIRClassType::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<MindIRClassType>(), std::make_shared<TypeType>());
}

abstract::AbstractBasePtr MindIRNameSpace::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<MindIRNameSpace>(), std::make_shared<External>());
}

abstract::AbstractBasePtr MindIRSymbol::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<MindIRSymbol>(), std::make_shared<External>());
}

abstract::AbstractBasePtr MindIRMetaFuncGraph::ToAbstract() {
  return std::make_shared<abstract::MetaFuncGraphAbstractClosure>(std::make_shared<MetaFuncGraph>(name()));
}
}  // namespace mindspore
