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

#include "infer/ops_func_impl/non_zero.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"

namespace mindspore::ops {

class NonZeroFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    const auto &x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto x_rank = SizeToLong(x_shape.size());

    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto jit_level = context_ptr->GetJitLevel();
    auto execution_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
    // GE mode circumvention: 910A and 910B have different results and different errors.
    if (x_shape.size() < kDim1 && execution_mode == kGraphMode && jit_level == "O2") {
      MS_EXCEPTION(RuntimeError)
        << "For 'Nonzero' op, the dimension of 'x' must be greater than or equal to 1, but got " << x_rank << ".";
    }
    if (IsDynamicRank(x_shape)) {
      x_rank = abstract::Shape::kShapeDimAny;
    }
    auto output_shape = ShapeVector({abstract::Shape::kShapeDimAny, x_rank});
    return std::make_shared<abstract::AbstractTensor>(kInt64, output_shape);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("NonZero", NonZeroFrontendFuncImpl);
}  // namespace mindspore::ops
