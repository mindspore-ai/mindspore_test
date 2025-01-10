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

#include <vector>
#include <memory>
#include "op_def/op_name.h"
#include "infer/ops_func_impl/prelu_grad.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/ms_context.h"

namespace mindspore::ops {
BaseShapePtr PReLUGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex1]->GetShape();
  auto w_shape = input_args[kIndex2]->GetShape();

  auto x_shape_vector = x_shape->GetShapeVector();
  if (!IsDynamicRank(x_shape_vector)) {
    auto x_rank = x_shape_vector.size();
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    int execution_mode = context->get_param<int>(MS_CTX_EXECUTION_MODE);
    bool is_ascend = context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice;

    if (is_ascend && x_rank <= 1 && execution_mode == kGraphMode && !context->IsKByKExecutorMode()) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', the dimension of 'x' can not be 0-D or 1-D when the platform is \"Ascend\", but got dimension of 'x' is "
        << x_rank << ".";
    }
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, w_shape});
}

TypePtr PReLUGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex1]->GetType();
  auto w_type = input_args[kIndex2]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, w_type});
}

TypePtrList PReLUGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &w_tensor = input_values[kIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(w_tensor);
  return {x_tensor->Dtype(), w_tensor->Dtype()};
}

ShapeArray PReLUGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &w_tensor = input_values[kIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(w_tensor);
  return {x_tensor->shape(), w_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNamePReLUGrad, PReLUGradFuncImpl)
}  // namespace mindspore::ops
