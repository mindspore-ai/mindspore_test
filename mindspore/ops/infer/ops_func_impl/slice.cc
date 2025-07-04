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
#include <set>
#include <memory>
#include <map>

#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "infer/ops_func_impl/slice.h"

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore::ops {
std::vector<int64_t> SliceFuncImpl::InferImplSliceFuncCalInputValue(const PrimitivePtr &primitive,
                                                                    const AbstractBasePtr &input_value) const {
  MS_EXCEPTION_IF_NULL(input_value);
  if (auto value_ptr = input_value->GetValue(); value_ptr == nullptr || !IsValueKnown(value_ptr)) {
    MS_EXCEPTION(TypeError) << "For Slice, currently, it is not "
                               "supported when 'begin' and/or 'size' has unknown value(s).";
  }
  std::vector<int64_t> tmp_input;
  if (CheckAndConvertUtils::IsTensor(input_value)) {
    auto input_value_type = input_value->GetType();
    tmp_input = CheckAndConvertUtils::CheckTensorIntValue("slice args value", input_value->GetValue(),
                                                          primitive->name(), input_value_type);
  } else if (CheckAndConvertUtils::IsTuple(input_value)) {
    tmp_input = CheckAndConvertUtils::CheckTupleInt("slice args value", input_value->GetValue(), primitive->name());
  } else if (CheckAndConvertUtils::IsList(input_value)) {
    tmp_input = CheckAndConvertUtils::CheckListInt("slice args value", input_value->GetValue(), primitive->name());
  } else {
    MS_EXCEPTION(TypeError) << "For Slice, the 'begin' and 'size' must be Tuple or List.";
  }

  return tmp_input;
}

BaseShapePtr SliceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  constexpr size_t kSliceInputNum = 3;
  MS_EXCEPTION_IF_CHECK_FAIL((input_args.size() == kSliceInputNum || input_args.size() == kSliceInputNum + 1),
                             "Slice inputs num error");
  auto input_x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape());
  auto input_x_shape = input_x_shape_map[kShape];
  auto input_begin = input_args[kInputIndex1];
  auto input_begin_value_ptr = input_begin->GetValue();
  MS_EXCEPTION_IF_NULL(input_begin_value_ptr);
  auto input_size = input_args[kInputIndex2];
  auto input_size_value_ptr = input_size->GetValue();
  MS_EXCEPTION_IF_NULL(input_size_value_ptr);
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  bool is_inputx_dyn = IsDynamic(input_x_shape);
  ShapeVector out_shape = {};
  if (input_x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Slice, the input_x must hava value.";
  }

  if (!IsValueKnown(input_begin_value_ptr) && IsValueKnown(input_size_value_ptr)) {
    auto tmp_input = InferImplSliceFuncCalInputValue(primitive, input_size);
    for (size_t i = 0; i < tmp_input.size(); i++) {
      out_shape.push_back(-1);
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }

  if (!IsValueKnown(input_size_value_ptr)) {
    auto arg = input_args[kInputIndex2];
    if (CheckAndConvertUtils::IsTensor(arg)) {
      auto tensor_shape = arg->GetShape()->GetShapeVector();
      if (tensor_shape.size() != 1) {
        MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
      }
    }
    out_shape = GetShapeValue(primitive, input_args[kInputIndex2]);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  auto input_begin_value = InferImplSliceFuncCalInputValue(primitive, input_begin);
  auto input_size_value = InferImplSliceFuncCalInputValue(primitive, input_size);
  auto rank = input_x_shape.size();
  if ((!is_inputx_dyn) && ((input_begin_value.size() != rank) || (input_size_value.size() != rank))) {
    MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("input_begin", input_begin_value, prim_name);
  for (size_t i = 0; i < rank; ++i) {
    if (input_x_shape[i] < 0) {
      continue;
    }
    if (input_size_value[i] < -1) {
      MS_EXCEPTION(RuntimeError) << "For Slice, the value in size should not be less than -1, but got "
                                 << input_size_value[i];
    }

    if (input_begin_value[i] + input_size_value[i] > input_x_shape[i]) {
      MS_EXCEPTION(ValueError) << "For Slice, the sum of begin_shape[" << i << "] and size_shape[" << i
                               << "] must be no greater than input_x_shape[" << i << "].";
    }
    if (input_size_value[i] == -1) {
      input_size_value[i] = input_x_shape[i] - input_begin_value[i];
    }
  }
  return std::make_shared<abstract::Shape>(input_size_value);
}

TypePtr SliceFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  return x_type;
}
}  // namespace mindspore::ops
