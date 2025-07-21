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

#include <memory>
#include <set>
#include <string>
#include <optional>
#include "infer/ops_func_impl/normal_float_float.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "ops_utils/type_dispatch.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void GetStdValue(const ValuePtr &std, std::optional<float> *std_opt) {
  std::optional<T> value = GetScalarValue<T>(std);
  if (value.has_value()) {
    *std_opt = static_cast<float>(value.value());
  }
}
}  // namespace

void NormalStdCheck(const PrimitivePtr &primitive, const AbstractBasePtr &std_abs) {
  if (CheckAndConvertUtils::IsTensor(std_abs)) {
    return;
  }

  std::optional<float> std_opt;
  TYPE_DISPATCH_PYTHON_NUMBER(std_abs->GetType()->type_id(), "GetStdValue",
                              [&]() { GetStdValue<scalar_t>(std_abs->GetValue(), &std_opt); });
  if (std_opt.has_value()) {
    auto std = std_opt.value();
    MS_CHECK_VALUE(std >= 0.0, CheckAndConvertUtils::FormatCheckIntegerMsg("std", std, kGreaterEqual, 0.0, primitive));
  }
}

BaseShapePtr NormalFloatFloatFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  if (!CheckAndConvertUtils::IsTensor(input_args[kInputIndex0]) &&
      !CheckAndConvertUtils::IsTensor(input_args[kInputIndex1])) {
    NormalStdCheck(primitive, input_args[kInputIndex1]);

    auto shape_shape = input_args[kInputIndex2]->GetShape();
    if (shape_shape->isa<abstract::DynamicSequenceShape>()) {
      return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
    auto shape_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
    if (!shape_array_opt.has_value()) {
      if (shape_shape->isa<abstract::SequenceShape>()) {
        auto seq_shape = shape_shape->cast<abstract::SequenceShapePtr>();
        MS_EXCEPTION_IF_NULL(seq_shape);
        size_t shape_size = seq_shape->size();
        return std::make_shared<abstract::TensorShape>(ShapeVector(shape_size, abstract::Shape::kShapeDimAny));
      }
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    auto shape_array = shape_array_opt.value();
    if (!shape_array.HasUnknownValue()) {
      std::vector<int64_t> shape_vec = shape_array.ToVector();
      if (std::any_of(shape_vec.begin(), shape_vec.end(), [](const int &shape_i) { return shape_i < 0; })) {
        MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                                 << "], the component of shape can't be less than 0, but got " << shape_vec;
      }
      auto x_shape_ptr = std::make_shared<abstract::Shape>(shape_vec);
      return x_shape_ptr;
    } else {
      ShapeVector output_shape;
      for (size_t i = 0; i < shape_array.size(); i++) {
        if (shape_array.IsValueUnknown(i)) {
          output_shape.push_back(abstract::Shape::kShapeDimAny);
        } else {
          output_shape.push_back(shape_array[i]);
        }
      }
      return std::make_shared<abstract::Shape>(output_shape);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', mean and std must be float, but got: " << input_args[kInputIndex0]->ToString()
                            << " and, " << input_args[kInputIndex1]->ToString() << ".";
  }
}

TypePtr NormalFloatFloatFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kFloat32);
}
}  // namespace ops
}  // namespace mindspore
