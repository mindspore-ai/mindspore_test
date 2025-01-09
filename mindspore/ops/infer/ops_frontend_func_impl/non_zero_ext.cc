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

#include "infer/ops_func_impl/non_zero_ext.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
class NonZeroExtFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    if (IsDynamicRank(x_shape)) {
      abstract::AbstractBasePtrList out_tensors = {
        std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector{abstract::TensorShape::kShapeRankAny})};

      auto out_tuple = std::make_shared<abstract::AbstractTuple>(out_tensors);
      out_tuple->CheckAndConvertToDynamicLenSequence();
      return out_tuple;
    }
    abstract::AbstractBasePtrList abs_list{};
    auto output_shape = ShapeVector({abstract::Shape::kShapeDimAny});
    // When the input dimension is 0d, the output is a tuple (out,) and the output dimension is 1d.
    auto x_rank = SizeToLong(x_shape.size()) == 0 ? 1 : SizeToLong(x_shape.size());
    for (int i = 0; i < x_rank; i++) {
      abs_list.push_back(std::make_shared<abstract::AbstractTensor>(kInt64, output_shape));
    }
    auto abs = std::make_shared<abstract::AbstractTuple>(abs_list);
    return abs;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("NonZeroExt", NonZeroExtFrontendFuncImpl);
}  // namespace mindspore::ops
