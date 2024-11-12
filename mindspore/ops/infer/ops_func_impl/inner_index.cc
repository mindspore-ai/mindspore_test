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

#include <functional>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <utility>
#include "infer/ops_func_impl/inner_index.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

TypePtr InnerIndexFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_LOG(EXCEPTION) << "Currently, the 'Index' supports only the pynative mode.";
  return input_args[kIndex0]->GetType();
}

TypePtrList InnerIndexFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return IndexFuncImpl::InferType(primitive, input_values);
}

ShapeArray InnerIndexFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return IndexFuncImpl::InferShape(primitive, input_values);
}

REGISTER_SIMPLE_INFER(kNameInnerIndex, InnerIndexFuncImpl)
}  // namespace ops
}  // namespace mindspore
