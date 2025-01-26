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

#ifndef TESTS_UT_CPP_OPERATOR_META_DSL_DENSE_H_
#define TESTS_UT_CPP_OPERATOR_META_DSL_DENSE_H_

#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
void CheckDenseInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args);
REGISTER_FUNCTION_OP(Dense, CheckDenseInputs);
}  // namespace mindspore::prim
#endif  // TESTS_UT_CPP_OPERATOR_META_DSL_DENSE_H_
