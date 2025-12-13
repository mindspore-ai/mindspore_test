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

#include <complex>
#include <map>
#include <memory>
#include <vector>
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"
#include "ir/tensor_new.h"
#include "abstract/utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename IN1_T, typename IN2_T, typename OUT_T>
void MulImpl(void *x1, void *x2, void *out, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(out);
  auto x1_data = static_cast<IN1_T *>(x1);
  auto x2_data = static_cast<IN2_T *>(x2);
  auto out_data = static_cast<OUT_T *>(out);

  for (size_t i = 0; i < size; ++i) {
    auto x1_value = static_cast<OUT_T>(x1_data[i]);
    auto x2_value = static_cast<OUT_T>(x2_data[i]);
    if constexpr (std::is_same_v<OUT_T, bool>) {
      out_data[i] = x1_value && x2_value;
    } else {
      out_data[i] = x1_value * x2_value;
    }
  }
}

#define MUL_CASE(SHORTHAND, TYPE, TYPE_C)                                                      \
  case kNumberType##SHORTHAND: {                                                               \
    MulImpl<T1, TYPE_C, T3>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), \
                            result_tensor->DataSize());                                        \
    break;                                                                                     \
  }

template <typename T1, typename T3>
void MulInnerDispatch(const tensor::TensorPtr &x1_tensor, const tensor::TensorPtr &x2_tensor,
                      const tensor::TensorPtr &result_tensor) {
  auto type_id2 = x2_tensor->data_type();
  switch (type_id2) {
    MUL_CASE(Float16, Float16, float16)
    MUL_CASE(Float32, Float32, float)
    MUL_CASE(Float64, Float64, double)
    MUL_CASE(Int8, Int8, int8_t)
    MUL_CASE(Int16, Int16, int16_t)
    MUL_CASE(Int32, Int32, int32_t)
    MUL_CASE(Int64, Int64, int64_t)
    MUL_CASE(UInt8, UInt8, uint8_t)
    MUL_CASE(BFloat16, BFloat16, bfloat16)
    MUL_CASE(Bool, Bool, bool)
    default:
      MS_LOG(EXCEPTION) << "For Mul, the second input data type is not supported.";
  }
}

#undef MUL_CASE

ValuePtr MulInferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x1_value = input_args[kIndex0]->GetValue();
  auto x2_value = input_args[kIndex1]->GetValue();
  if (x1_value == nullptr || x2_value == nullptr || x1_value->isa<ValueAny>() || x2_value->isa<ValueAny>()) {
    return nullptr;
  }
  auto x1_tensor = x1_value->cast<tensor::TensorPtr>();
  auto x2_tensor = x2_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  MS_EXCEPTION_IF_NULL(x2_tensor);

  auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto x2_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (IsDynamic(x1_shape) || IsDynamic(x2_shape) || !IsMactchedShapeInferValue(x1_shape, x2_shape)) {
    return nullptr;
  }

  auto type_id1 = x1_tensor->data_type();
  auto type_id2 = x2_tensor->data_type();
  auto out_type_id = PromoteType(type_id1, type_id2, primitive->name());

  auto out_shape = x1_shape.size() > x2_shape.size() ? x1_shape : x2_shape;
  auto result_tensor = std::make_shared<tensor::Tensor>(out_type_id, out_shape);

#define MUL_DISPATCH(IN_TYPE, IN_TYPE_C)                                            \
  case kNumberType##IN_TYPE: {                                                      \
    switch (PromoteType(type_id1, type_id2, "")) {                                  \
      case kNumberTypeFloat16:                                                      \
        MulInnerDispatch<IN_TYPE_C, float16>(x1_tensor, x2_tensor, result_tensor);  \
        break;                                                                      \
      case kNumberTypeFloat32:                                                      \
        MulInnerDispatch<IN_TYPE_C, float>(x1_tensor, x2_tensor, result_tensor);    \
        break;                                                                      \
      case kNumberTypeFloat64:                                                      \
        MulInnerDispatch<IN_TYPE_C, double>(x1_tensor, x2_tensor, result_tensor);   \
        break;                                                                      \
      case kNumberTypeInt8:                                                         \
        MulInnerDispatch<IN_TYPE_C, int8_t>(x1_tensor, x2_tensor, result_tensor);   \
        break;                                                                      \
      case kNumberTypeInt16:                                                        \
        MulInnerDispatch<IN_TYPE_C, int16_t>(x1_tensor, x2_tensor, result_tensor);  \
        break;                                                                      \
      case kNumberTypeInt32:                                                        \
        MulInnerDispatch<IN_TYPE_C, int32_t>(x1_tensor, x2_tensor, result_tensor);  \
        break;                                                                      \
      case kNumberTypeInt64:                                                        \
        MulInnerDispatch<IN_TYPE_C, int64_t>(x1_tensor, x2_tensor, result_tensor);  \
        break;                                                                      \
      case kNumberTypeUInt8:                                                        \
        MulInnerDispatch<IN_TYPE_C, uint8_t>(x1_tensor, x2_tensor, result_tensor);  \
        break;                                                                      \
      case kNumberTypeBFloat16:                                                     \
        MulInnerDispatch<IN_TYPE_C, bfloat16>(x1_tensor, x2_tensor, result_tensor); \
        break;                                                                      \
      case kNumberTypeBool:                                                         \
        MulInnerDispatch<IN_TYPE_C, bool>(x1_tensor, x2_tensor, result_tensor);     \
        break;                                                                      \
      default:                                                                      \
        MS_LOG(EXCEPTION) << "For Mul, the output data type is not supported.";     \
    }                                                                               \
    break;                                                                          \
  }

  switch (type_id1) {
    MUL_DISPATCH(Float16, float16)
    MUL_DISPATCH(Float32, float)
    MUL_DISPATCH(Float64, double)
    MUL_DISPATCH(Int8, int8_t)
    MUL_DISPATCH(Int16, int16_t)
    MUL_DISPATCH(Int32, int32_t)
    MUL_DISPATCH(Int64, int64_t)
    MUL_DISPATCH(UInt8, uint8_t)
    MUL_DISPATCH(BFloat16, bfloat16)
    MUL_DISPATCH(Bool, bool)
    default:
      MS_LOG(DEBUG) << "For '" << primitive->name()
                    << "', the inputs data type is not supported now. Take it as constant folding fallback.";
      return nullptr;
  }
#undef MUL_DISPATCH
  return result_tensor;
}
}  // namespace

class MulFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MulInferValue(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Mul", MulFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
