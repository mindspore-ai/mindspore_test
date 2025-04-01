/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_

#include <string>
#include <map>
#include <set>
#include <memory>
#include "utils/ms_utils.h"
#include "include/common/visible.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ir/meta_func_graph.h"
#include "ir/core_ops_primitive.h"

namespace mindspore::prim {
#define EXPAND_PARAMS(...) __VA_ARGS__

#define DECLARE_PARAM(param) NodePtr param = NewParam(#param);

#define DECLARE_PARAMS_0()

#define DECLARE_PARAMS_1(p1) DECLARE_PARAM(p1)

#define DECLARE_PARAMS_2(p1, p2) \
  DECLARE_PARAM(p1);             \
  DECLARE_PARAM(p2)

#define DECLARE_PARAMS_3(p1, p2, p3) \
  DECLARE_PARAMS_2(p1, p2);          \
  DECLARE_PARAM(p3);

#define DECLARE_PARAMS_4(p1, p2, p3, p4) \
  DECLARE_PARAMS_3(p1, p2, p3)           \
  DECLARE_PARAM(p4);

#define DECLARE_PARAMS_5(p1, p2, p3, p4, p5) \
  DECLARE_PARAMS_4(p1, p2, p3, p4);          \
  DECLARE_PARAM(p5)

#define DECLARE_PARAMS_6(p1, p2, p3, p4, p5, p6) \
  DECLARE_PARAMS_5(p1, p2, p3, p4, p5);          \
  DECLARE_PARAM(p6)

#define DECLARE_PARAMS_7(p1, p2, p3, p4, p5, p6, p7) \
  DECLARE_PARAMS_6(p1, p2, p3, p4, p5, p6);          \
  DECLARE_PARAM(p7)

#define DECLARE_PARAMS_8(p1, p2, p3, p4, p5, p6, p7, p8) \
  DECLARE_PARAMS_7(p1, p2, p3, p4, p5, p6, p7);          \
  DECLARE_PARAM(p8)

#define DECLARE_PARAMS_9(p1, p2, p3, p4, p5, p6, p7, p8, p9) \
  DECLARE_PARAMS_8(p1, p2, p3, p4, p5, p6, p7, p8);          \
  DECLARE_PARAM(p9)

#define DECLARE_PARAMS_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10) \
  DECLARE_PARAMS_9(p1, p2, p3, p4, p5, p6, p7, p8, p9);            \
  DECLARE_PARAM(p10)

#define DECLARE_PARAMS_11(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) \
  DECLARE_PARAMS_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);           \
  DECLARE_PARAM(p11)

#define DECLARE_PARAMS_12(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) \
  DECLARE_PARAMS_11(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);           \
  DECLARE_PARAM(p12)

#define DECLARE_PARAMS_13(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13) \
  DECLARE_PARAMS_12(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);           \
  DECLARE_PARAM(p13)

#define DECLARE_PARAMS_14(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14) \
  DECLARE_PARAMS_13(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);           \
  DECLARE_PARAM(p14)

#define DECLARE_PARAMS_15(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15) \
  DECLARE_PARAMS_14(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);           \
  DECLARE_PARAM(p15)

#define DECLARE_PARAMS_16(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16) \
  DECLARE_PARAMS_15(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);           \
  DECLARE_PARAM(p16)

#define DECLARE_PARAMS_17(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17) \
  DECLARE_PARAMS_16(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);           \
  DECLARE_PARAM(p17)

#define DECLARE_PARAMS_18(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18) \
  DECLARE_PARAMS_17(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);           \
  DECLARE_PARAM(p18)

#define DECLARE_PARAMS_19(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19) \
  DECLARE_PARAMS_18(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);           \
  DECLARE_PARAM(p19)

#define DECLARE_PARAMS_20(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20) \
  DECLARE_PARAMS_19(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);           \
  DECLARE_PARAM(p20)

#define GET_DECLARE_PARAMS_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                                 _20, NAME, ...)                                                                       \
  NAME

// DECLARE_PARAMS support 1 to 20 parameters.
#define DECLARE_PARAMS(...)                                                                                           \
  _EXPAND(GET_DECLARE_PARAMS_MACRO(                                                                                   \
    __VA_ARGS__, DECLARE_PARAMS_20, DECLARE_PARAMS_19, DECLARE_PARAMS_18, DECLARE_PARAMS_17, DECLARE_PARAMS_16,       \
    DECLARE_PARAMS_15, DECLARE_PARAMS_14, DECLARE_PARAMS_13, DECLARE_PARAMS_12, DECLARE_PARAMS_11, DECLARE_PARAMS_10, \
    DECLARE_PARAMS_9, DECLARE_PARAMS_8, DECLARE_PARAMS_7, DECLARE_PARAMS_6, DECLARE_PARAMS_5, DECLARE_PARAMS_4,       \
    DECLARE_PARAMS_3, DECLARE_PARAMS_2, DECLARE_PARAMS_1, DECLARE_PARAMS_0)(__VA_ARGS__))

// There are variable length elements in params.
#define IF_IMPL_0(cond, true_case, false_case, ...) \
  IfCond(                                           \
    cond,                                           \
    [this, true_case]() {                           \
      DECLARE_PARAMS(__VA_ARGS__);                  \
      true_case();                                  \
    },                                              \
    [this, false_case]() {                          \
      DECLARE_PARAMS(__VA_ARGS__);                  \
      false_case();                                 \
    },                                              \
    {__VA_ARGS__})

// The params is ().
#define IF_IMPL_1(cond, true_case, false_case, ...) \
  IfCond(                                           \
    cond, [this, true_case]() { true_case(); }, [this, false_case]() { false_case(); }, {})

// Select different macro definitions depending on whether params is empty.
#define IF_IMPL_DISPATCH(cond, true_case, false_case, is_empty, ...) \
  IF_IMPL_##is_empty(cond, true_case, false_case, __VA_ARGS__)

#define IF_IMPL_SELECT(cond, true_case, false_case, is_empty, ...) \
  IF_IMPL_DISPATCH(cond, true_case, false_case, is_empty, __VA_ARGS__)

#define ARG_N(_1, _2, N, ...) N

#define IS_EMPTY(params) ARG_N(EXPAND_PARAMS params, 0, 1)

// Define IF_IMPL.
#define IF_IMPL(cond, true_case, false_case, params) \
  IF_IMPL_SELECT(cond, true_case, false_case, IS_EMPTY(params), EXPAND_PARAMS params)

// Definition of MetaImpl subclass.
#define _DEFINE_FUNCTION_OP(name, check_func, bprop_func) \
  class name##MetaImpl : public MetaImpl {                \
   public:                                                \
    explicit name##MetaImpl() : MetaImpl(#name) {         \
      set_check_func(check_func);                         \
      set_bprop_func(bprop_func);                         \
    }                                                     \
    ~name##MetaImpl() override = default;                 \
    MS_DECLARE_PARENT(name##MetaImpl, MetaImpl)           \
    void GenerateFunction() override;                     \
  };                                                      \
  static const MetaImplRegHelper meta_impl_helper_##name(#name, []() { return std::make_shared<name##MetaImpl>(); });

// DEFINE_FUNCTION_OP(op_name) -> _DEFINE_FUNCTION_OP_1
#define _DEFINE_FUNCTION_OP_1(name) _DEFINE_FUNCTION_OP(name, nullptr, nullptr)

// DEFINE_FUNCTION_OP(op_name, check_func) -> _DEFINE_FUNCTION_OP_2
#define _DEFINE_FUNCTION_OP_2(name, check_func) _DEFINE_FUNCTION_OP(name, check_func, nullptr)

// DEFINE_FUNCTION_OP(op_name, check_func, bprop) -> _DEFINE_FUNCTION_OP_3

#define _DEFINE_FUNCTION_OP_3(name, check_func, bprop) \
  _DEFINE_FUNCTION_OP(bprop, nullptr, nullptr)         \
  _DEFINE_FUNCTION_OP(name, check_func, []() { return std::make_shared<bprop##MetaImpl>(); })

#define _EXPAND(x) x
#define _GET_FUNCTION_OP_MACRO(_1, _2, _3, NAME, ...) NAME

// Define REGISTER_FUNCTION_OP api.
#define REGISTER_FUNCTION_OP(...)                                                           \
  _EXPAND(_GET_FUNCTION_OP_MACRO(__VA_ARGS__, _DEFINE_FUNCTION_OP_3, _DEFINE_FUNCTION_OP_2, \
                                 _DEFINE_FUNCTION_OP_1)(__VA_ARGS__))

#define BeginFunction(name, ...)            \
  void name##MetaImpl::GenerateFunction() { \
    do {                                    \
    _EXPAND(DECLARE_PARAMS(__VA_ARGS__))

#define EndFunction(name)       \
  /* Used with BeginFunction */ \
  }                             \
  while (0)                     \
    ;                           \
  }
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_
