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

#include "infer/ops_func_impl/batch_norm_elemt_grad.h"
#include <memory>
#include "mindapi/base/types.h"
#include "abstract/dshape.h"
#include "op_def/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormElemtGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex1]->GetShape();
  return input_shape->Clone();
}

TypePtr BatchNormElemtGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex1]->GetType();
  return x_type->Clone();
}

}  // namespace ops
}  // namespace mindspore
