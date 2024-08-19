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
#include <memory>
#include "infer/ops_func_impl/trace_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {

OP_FUNC_IMPL_INFER_TEST_DECLARE(TraceExt, EltwiseOpParams);

OP_FUNC_IMPL_INFER_TEST_CASES(TraceExt, testing::Values(EltwiseOpParams{{3, 3}, kFloat32, {}, kFloat32, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 5}, kFloat32, {}, kFloat32, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{2, 1}, kFloat32, {}, kFloat32, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{-1, -1}, kFloat32, {}, kFloat32, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{-2}, kFloat32, {}, kFloat32, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kBool, {}, kInt64, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kInt8, {}, kInt64, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kInt16, {}, kInt64, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kInt32, {}, kInt64, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kInt64, {}, kInt64, {CreateScalar(kValueAny)}},
                                                  EltwiseOpParams{{3, 3}, kUInt8, {}, kInt64, {CreateScalar(kValueAny)}}));
}  // namespace mindspore::ops
