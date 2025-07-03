/*
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

#ifndef TESTS_UT_CPP_OPERATOR_META_DSL_UNIT_TEST_API_DEFINE_H_
#define TESTS_UT_CPP_OPERATOR_META_DSL_UNIT_TEST_API_DEFINE_H_

#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
REGISTER_FUNCTION_OP(TestIsInstance);
REGISTER_FUNCTION_OP(TestIf);
REGISTER_FUNCTION_OP(TestIfExp, nullptr);
REGISTER_FUNCTION_OP(TestFor);
REGISTER_FUNCTION_OP(TestForiLoop);
REGISTER_FUNCTION_OP(TestWhile);
REGISTER_FUNCTION_OP(TestScan);
REGISTER_FUNCTION_OP(TestAnd);
REGISTER_FUNCTION_OP(TestOr);
REGISTER_FUNCTION_OP(TestDtype);
REGISTER_FUNCTION_OP(TestAllAny);
}  // namespace mindspore::prim
#endif  // TESTS_UT_CPP_OPERATOR_META_DSL_UNIT_TEST_API_DEFINE_H_
