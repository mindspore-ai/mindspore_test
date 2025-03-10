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

#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {

class SplitTensorFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    auto input_abs = input_args[kIndex0];
    auto input_shape_ptr = input_abs->GetShape();
    auto input_shape = input_shape_ptr->GetShapeVector();
    auto axis_value = input_args[kIndex2]->GetValue();
    auto split_size_value = input_args[kIndex1]->GetValue();
    AbstractBasePtrList output_list{};
    auto axis_opt = GetScalarValue<int64_t>(axis_value);
    auto split_sections_opt = GetScalarValue<int64_t>(split_size_value);
    if (!axis_opt.has_value() || !split_sections_opt.has_value() || IsDynamicRank(input_shape) ||
        input_shape[(axis_opt.value() + static_cast<int64_t>(input_shape.size())) %
                    static_cast<int64_t>(input_shape.size())] == abstract::Shape::kShapeDimAny) {
      MS_EXCEPTION(ValueError) << "Unsupported dynamic shape for graph mode.";
      auto dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      output_list.push_back(abstract::MakeAbstractTensor(dynamic_shape, input_abs->GetType()));
      auto abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
      abs_tuple->CheckAndConvertToDynamicLenSequence();
      return abs_tuple;
    }
    auto rank = SizeToLong(input_shape.size());
    if (rank <= 0) {
      MS_EXCEPTION(ValueError) << "For [SplitWithSize], expected at least a 1-dimensional tensor.";
    }
    auto dim = axis_opt.value();
    auto split_size = split_sections_opt.value();
    if (split_size == 0) {
      MS_EXCEPTION(ValueError) << "split_size's value cannot be zero";
    }
    if (dim < 0) {
      dim += rank;
    }
    size_t pos = LongToSize(dim);
    int64_t split_num = input_shape[pos] / split_size;
    int64_t remaining = input_shape[pos] % split_size;
    std::vector<int64_t> output_shape = input_shape;
    for (int64_t i = 0; i < split_num; ++i) {
      output_shape[pos] = split_size;
      auto split_tensor_abstract =
        abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(output_shape), input_abs->GetType());
      (void)output_list.push_back(split_tensor_abstract);
    }
    if (remaining != 0) {
      output_shape[pos] = remaining;
      auto split_tensor_abstract =
        abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(output_shape), input_abs->GetType());
      (void)output_list.push_back(split_tensor_abstract);
    }
    auto abs = std::make_shared<abstract::AbstractTuple>(output_list);
    return abs;
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("SplitTensor", SplitTensorFrontendFuncImpl);

// Reuse abstract infer for SplitTensorView kernel
class SplitTensorViewFrontendFuncImpl : public SplitTensorFrontendFuncImpl {};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("SplitTensorView", SplitTensorViewFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
