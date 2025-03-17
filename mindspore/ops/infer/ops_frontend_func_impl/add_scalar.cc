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
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
bool isAddInputBoolType(TypeId t) { return t == kNumberTypeBool; }

bool isAddInputFloatType(TypeId t) {
  return t == kNumberTypeBFloat16 || t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64;
}

template <typename T, typename U>
void ImplAddScalarHelp(void *x1, float x2, float x3, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(result);
  U *x1_data = static_cast<U *>(x1);
  T x2_data = static_cast<T>(x2);
  T x3_data = static_cast<T>(x3);
  auto result_data = static_cast<T *>(result);
  for (size_t i = 0; i < size; ++i) {
    T x1_ele = static_cast<T>(x1_data[i]);
    result_data[i] = x1_ele + x2_data * x3_data;
  }
}

template <typename T>
void ImplAddScalarBoolHelp(void *x1, float x2, float x3, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T x2_data = static_cast<T>(x2);
  T x3_data = static_cast<T>(x3);
  auto result_data = static_cast<T *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = x1_data[i] + (x3_data * x2_data);
  }
}

template <typename T>
void ImplAddScalar(void *x1, TypeId x1_type, float x2, float x3, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeBool:
      ImplAddScalarHelp<T, bool>(x1, x2, x3, result, size);
      break;
    case kNumberTypeInt8:
      ImplAddScalarHelp<T, int8_t>(x1, x2, x3, result, size);
      break;
    case kNumberTypeInt16:
      ImplAddScalarHelp<T, int16_t>(x1, x2, x3, result, size);
      break;
    case kNumberTypeInt32:
      ImplAddScalarHelp<T, int32_t>(x1, x2, x3, result, size);
      break;
    case kNumberTypeInt64:
      ImplAddScalarHelp<T, int64_t>(x1, x2, x3, result, size);
      break;
    case kNumberTypeUInt8:
      ImplAddScalarHelp<T, uint8_t>(x1, x2, x3, result, size);
      break;
    case kNumberTypeFloat16:
      ImplAddScalarHelp<T, float16>(x1, x2, x3, result, size);
      break;
    case kNumberTypeBFloat16:
      ImplAddScalarHelp<T, bfloat16>(x1, x2, x3, result, size);
      break;
    case kNumberTypeFloat32:
      ImplAddScalarHelp<T, float>(x1, x2, x3, result, size);
      break;
    case kNumberTypeFloat64:
      ImplAddScalarHelp<T, double>(x1, x2, x3, result, size);
      break;
    default:
      break;
  }
}

template <typename T>
void ImplAddScalarComplex(void *x1, TypeId x1_type, float x2, float x3, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeComplex64:
      ImplAddScalarHelp<T, std::complex<float>>(x1, x2, x3, result, size);
      break;
    case kNumberTypeComplex128:
      ImplAddScalarHelp<T, std::complex<double>>(x1, x2, x3, result, size);
      break;
    default:
      break;
  }
}

template <typename T>
void ImplAddScalarBool(void *x1, TypeId x1_type, float x2, float x3, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeBool:
      ImplAddScalarBoolHelp<bool>(x1, x2, x3, result, size);
      break;
    default:
      break;
  }
}

using AddsHandler = std::function<void(void *x1, TypeId x1_type, float x2, float x3, void *result, size_t size)>;
std::map<TypeId, AddsHandler> add_scalar_impl_list = {
  {kNumberTypeBool, ImplAddScalarBool<bool>},
  {kNumberTypeInt8, ImplAddScalar<int8_t>},
  {kNumberTypeInt16, ImplAddScalar<int16_t>},
  {kNumberTypeInt32, ImplAddScalar<int32_t>},
  {kNumberTypeInt64, ImplAddScalar<int64_t>},
  {kNumberTypeUInt8, ImplAddScalar<uint8_t>},
  {kNumberTypeFloat16, ImplAddScalar<float16>},
  {kNumberTypeBFloat16, ImplAddScalar<bfloat16>},
  {kNumberTypeFloat32, ImplAddScalar<float>},
  {kNumberTypeFloat64, ImplAddScalar<double>},
  {kNumberTypeComplex64, ImplAddScalarComplex<std::complex<float>>},
  {kNumberTypeComplex128, ImplAddScalarComplex<std::complex<double>>}};

class AddScalarFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x1 = input_args[kIndex0]->GetValue();
    auto x2 = input_args[kIndex1]->GetValue();
    auto x3 = input_args[kIndex2]->GetValue();
    if (x1 == nullptr || x2 == nullptr || x3 == nullptr || x1->isa<ValueAny>() || x2->isa<ValueAny>() ||
        x3->isa<ValueAny>()) {
      return nullptr;
    }
    auto x1_tensor = x1->cast<tensor::TensorPtr>();
    auto x2_number = GetScalarCastValue<float>("AddScalar", x2);
    auto x3_number = GetScalarCastValue<float>("AddScalar", x3);
    MS_EXCEPTION_IF_NULL(x1_tensor);

    auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x1_shape)) {
      return nullptr;
    }

    auto x1_type = x1_tensor->data_type();
    auto x2_type = input_args[kIndex1]->GetType()->type_id();
    auto x3_type = input_args[kIndex2]->GetType()->type_id();
    if (x3_type == kNumberTypeBool && !isAddInputBoolType(x1_type)) {
      MS_LOG(EXCEPTION) << "When alpha type is Bool, the input type should be Bool";
    }

    if (x3_type == kNumberTypeFloat32 && !isAddInputFloatType(x1_type)) {
      MS_LOG(EXCEPTION) << "When alpha type is Float, the input type should be Float";
    }

    auto data_size = x1_tensor->DataSize();
    TypeId common_type = ConvertTypeBetweenTensorAndScalar(x1_type, x2_type, GetHashId(x1_type, x2_type));
    auto result_tensor = std::make_shared<tensor::Tensor>(common_type, x1_shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto iter = add_scalar_impl_list.find(common_type);
    if (iter == add_scalar_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                    << ", the type is not supported.";
      return nullptr;
    }

    iter->second(x1_tensor->data_c(), x1_type, x2_number, x3_number, result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("AddScalar", AddScalarFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
