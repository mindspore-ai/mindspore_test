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

#include "common/common_test.h"
#include "infer/ops_func_impl/selu_ext.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_INFER_TEST_DECLARE(SeLUExt, EltwiseOpParams);
OP_FUNC_IMPL_INFER_TEST_CASES(SeLUExt, testing::Values(EltwiseOpParams{{3, 2, 3}, kFloat32, {3, 2, 3}, kFloat32},
                                                       EltwiseOpParams{{3, 2, 3}, kFloat16, {3, 2, 3}, kFloat16},
                                                       EltwiseOpParams{{3, 2, -1}, kBFloat16, {3, 2, -1}, kBFloat16}));
}  // namespace ops
}  // namespace mindspore