/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_OP_UTILS_TYPE_DISPATCH_H
#define MINDSPORE_CORE_OPS_OP_UTILS_TYPE_DISPATCH_H

#include <cstdint>
#include <complex>
#include "ir/dtype/type_id.h"
#include "base/float16.h"

namespace mindspore {
template <TypeId N>
struct MsTypeToCppType {};

#define REG_MS_TYPE_TO_CPP_TYPE(MS_TYPE, CPP_TYPE) \
  template <>                                      \
  struct MsTypeToCppType<MS_TYPE> {                \
    using type = CPP_TYPE;                         \
  }

REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeDouble, double);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeFloat64, double);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeFloat, float);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeFloat32, float);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeFloat16, float16);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeBFloat16, bfloat16);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeInt64, int64_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeInt32, int32_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeInt16, int16_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeInt8, int8_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeUInt64, uint64_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeUInt32, uint32_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeUInt16, uint16_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeUInt8, uint8_t);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeComplex128, std::complex<double>);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeComplex64, std::complex<float>);
REG_MS_TYPE_TO_CPP_TYPE(kNumberTypeBool, bool);

template <TypeId T>
using MS_TYPE_TO_CPP_TYPE = typename MsTypeToCppType<T>::type;

#define TYPE_DISPATCH(DTYPE, NAME, ...)                                                                          \
  [&] {                                                                                                          \
    const TypeId &_dtype = DTYPE;                                                                                \
    constexpr const char *_dispatch_name = NAME;                                                                 \
    switch (_dtype) {                                                                                            \
      __VA_ARGS__;                                                                                               \
      default:                                                                                                   \
        MS_EXCEPTION(TypeError) << "Unsupported type [" << TypeIdToString(_dtype) << "] for " << _dispatch_name; \
    }                                                                                                            \
  }()

#define TYPE_SWITCH_CASE(DTYPE, ...)               \
  case DTYPE:                                      \
    do {                                           \
      using scalar_t = MS_TYPE_TO_CPP_TYPE<DTYPE>; \
      return __VA_ARGS__();                        \
    } while (0);                                   \
    break

#define TYPE_SWITCH_COMPLEX_CASES(...)                  \
  TYPE_SWITCH_CASE(kNumberTypeComplex128, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeComplex64, __VA_ARGS__)

#define TYPE_SWITCH_BOOL_CASES(...) TYPE_SWITCH_CASE(kNumberTypeBool, __VA_ARGS__)

#define TYPE_SWITCH_FLOATING_CASES(...)              \
  TYPE_SWITCH_CASE(kNumberTypeDouble, __VA_ARGS__);  \
  TYPE_SWITCH_CASE(kNumberTypeFloat, __VA_ARGS__);   \
  TYPE_SWITCH_CASE(kNumberTypeFloat64, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeFloat32, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeFloat16, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeBFloat16, __VA_ARGS__)

#define TYPE_SWITCH_FLOATING_CASES_AND(EXTRA_TYPE, ...) \
  TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__);              \
  TYPE_SWITCH_CASE(EXTRA_TYPE, __VA_ARGS__)

#define TYPE_SWITCH_FLOATING_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, ...) \
  TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__);                             \
  TYPE_SWITCH_CASE(EXTRA_TYPE1, __VA_ARGS__);                          \
  TYPE_SWITCH_CASE(EXTRA_TYPE2, __VA_ARGS__)

#define TYPE_SWITCH_FLOATING_CASES_AND_COMPLEX(...) \
  TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__);          \
  TYPE_SWITCH_COMPLEX_CASES(__VA_ARGS__)

#define TYPE_SWITCH_SIGNED_INT_CASES(...)          \
  TYPE_SWITCH_CASE(kNumberTypeInt8, __VA_ARGS__);  \
  TYPE_SWITCH_CASE(kNumberTypeInt16, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeInt32, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeInt64, __VA_ARGS__)

#define TYPE_SWITCH_UNSIGNED_INT_CASES(...)         \
  TYPE_SWITCH_CASE(kNumberTypeUInt8, __VA_ARGS__);  \
  TYPE_SWITCH_CASE(kNumberTypeUInt16, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeUInt32, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeUInt64, __VA_ARGS__)

#define TYPE_SWITCH_PYTHON_NUMBER_CASES(...)         \
  TYPE_SWITCH_CASE(kNumberTypeBool, __VA_ARGS__);    \
  TYPE_SWITCH_CASE(kNumberTypeInt64, __VA_ARGS__);   \
  TYPE_SWITCH_CASE(kNumberTypeFloat32, __VA_ARGS__); \
  TYPE_SWITCH_CASE(kNumberTypeFloat64, __VA_ARGS__)

#define TYPE_SWITCH_INT_CASES(...)           \
  TYPE_SWITCH_SIGNED_INT_CASES(__VA_ARGS__); \
  TYPE_SWITCH_UNSIGNED_INT_CASES(__VA_ARGS__)

#define TYPE_SWITCH_INT_CASES_AND(EXTRA_TYPE, ...) \
  TYPE_SWITCH_INT_CASES(__VA_ARGS__);              \
  TYPE_SWITCH_CASE(EXTRA_TYPE, __VA_ARGS__)

#define TYPE_SWITCH_INT_CASES_AND_BOOL(...) TYPE_SWITCH_INT_CASES_AND(kNumberTypeBool, __VA_ARGS__)

#define TYPE_SWITCH_INT_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, ...) \
  TYPE_SWITCH_INT_CASES(__VA_ARGS__);                             \
  TYPE_SWITCH_CASE(EXTRA_TYPE1, __VA_ARGS__);                     \
  TYPE_SWITCH_CASE(EXTRA_TYPE2, __VA_ARGS__)

#define TYPE_SWITCH_FULL_INT_CASES(...)      \
  TYPE_SWITCH_SIGNED_INT_CASES(__VA_ARGS__); \
  TYPE_SWITCH_UNSIGNED_INT_CASES(__VA_ARGS__)

#define TYPE_SWITCH_FULL_INT_CASES_AND(EXTRA_TYPE, ...) \
  TYPE_SWITCH_FULL_INT_CASES(__VA_ARGS__);              \
  TYPE_SWITCH_CASE(EXTRA_TYPE, __VA_ARGS__)

#define TYPE_SWITCH_FULL_INT_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, ...) \
  TYPE_SWITCH_FULL_INT_CASES(__VA_ARGS__);                             \
  TYPE_SWITCH_CASE(EXTRA_TYPE1, __VA_ARGS__);                          \
  TYPE_SWITCH_CASE(EXTRA_TYPE2, __VA_ARGS__)

#define TYPE_SWITCH_ALL_CASES(...)    \
  TYPE_SWITCH_INT_CASES(__VA_ARGS__); \
  TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__)

#define TYPE_SWITCH_ALL_CASES_AND(EXTRA_TYPE, ...) \
  TYPE_SWITCH_ALL_CASES(__VA_ARGS__);              \
  TYPE_SWITCH_CASE(EXTRA_TYPE, __VA_ARGS__)

#define TYPE_SWITCH_ALL_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, ...) \
  TYPE_SWITCH_ALL_CASES(__VA_ARGS__);                             \
  TYPE_SWITCH_CASE(EXTRA_TYPE1, __VA_ARGS__);                     \
  TYPE_SWITCH_CASE(EXTRA_TYPE2, __VA_ARGS__)

#define TYPE_SWITCH_FULL_CASES(...)            \
  TYPE_SWITCH_BOOL_CASES(__VA_ARGS__);         \
  TYPE_SWITCH_COMPLEX_CASES(__VA_ARGS__);      \
  TYPE_SWITCH_SIGNED_INT_CASES(__VA_ARGS__);   \
  TYPE_SWITCH_UNSIGNED_INT_CASES(__VA_ARGS__); \
  TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__)

// floating + signed + unsigned
#define TYPE_DISPATCH_ALL(DTYPE, NAME, ...) TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_ALL_CASES(__VA_ARGS__))

#define TYPE_DISPATCH_ALL_AND(EXTRA_TYPE, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_ALL_CASES_AND(EXTRA_TYPE, __VA_ARGS__))

#define TYPE_DISPATCH_ALL_AND2(EXTRA_TYPE1, EXTRA_TYPE2, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_ALL_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, __VA_ARGS__))

// floating + signed + unsigned + complex + bool
#define TYPE_DISPATCH_FULL(DTYPE, NAME, ...) TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FULL_CASES(__VA_ARGS__))

#define TYPE_DISPATCH_INT(DTYPE, NAME, ...) TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FULL_INT_CASES(__VA_ARGS__))

#define TYPE_DISPATCH_INT_AND_BOOL(DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FULL_INT_CASES(__VA_ARGS__), TYPE_SWITCH_BOOL_CASES(__VA_ARGS__))

#define TYPE_DISPATCH_INT_AND(EXTRA_TYPE, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_INT_CASES_AND(EXTRA_TYPE, __VA_ARGS__))

#define TYPE_DISPATCH_INT_AND2(EXTRA_TYPE1, EXTRA_TYPE2, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_INT_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, __VA_ARGS__))

#define TYPE_DISPATCH_FLOATING(DTYPE, NAME, ...) TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FLOATING_CASES(__VA_ARGS__))

#define TYPE_DISPATCH_FLOATING_AND(EXTRA_TYPE, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FLOATING_CASES_AND(EXTRA_TYPE, __VA_ARGS__))

#define TYPE_DISPATCH_FLOATING_AND2(EXTRA_TYPE1, EXTRA_TYPE2, DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FLOATING_CASES_AND2(EXTRA_TYPE1, EXTRA_TYPE2, __VA_ARGS__))

#define TYPE_DISPATCH_FLOATING_AND_COMPLEX(DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_FLOATING_CASES_AND_COMPLEX(__VA_ARGS__))

#define TYPE_DISPATCH_PYTHON_NUMBER(DTYPE, NAME, ...) \
  TYPE_DISPATCH(DTYPE, NAME, TYPE_SWITCH_PYTHON_NUMBER_CASES(__VA_ARGS__))
}  // namespace mindspore
#endif
