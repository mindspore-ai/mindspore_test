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

#include "infer/ops_func_impl/tracev2_grad.h"

#include "op_def/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
BaseShapePtr TraceV2GradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_arg = input_args[1];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr TraceV2GradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto out_type = input_args[kInputIndex0]->GetType();
  return out_type;
}
}  // namespace mindspore::ops
