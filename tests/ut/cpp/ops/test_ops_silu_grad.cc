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
#include "common/common_test.h"
#include "infer/ops_func_impl/silu_grad.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
class TestSiLUGrad : public TestOps,
                     public testing::WithParamInterface<std::tuple<EltwiseGradOpShapeParams, EltwiseGradOpTypeParams>> {
};

TEST_P(TestSiLUGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto dy = std::make_shared<abstract::AbstractTensor>(dtype_param.grad_type, shape_param.grad_shape);
  ASSERT_NE(dy, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  DoFuncImplInferAndCompare<SiLUGradFuncImpl>("SiLUGrad", {dy, x}, expect_shape, expect_type);
}

namespace {
auto SiLUGradOpTypeCases = testing::ValuesIn({
  EltwiseGradOpTypeParams{kFloat16, kFloat16, kFloat16},
  EltwiseGradOpTypeParams{kFloat32, kFloat32, kFloat32},
  EltwiseGradOpTypeParams{kFloat64, kFloat64, kFloat64},
  EltwiseGradOpTypeParams{kComplex64, kComplex64, kComplex64},
  EltwiseGradOpTypeParams{kComplex128, kComplex128, kComplex128},
});
}

INSTANTIATE_TEST_CASE_P(TestSiLUGradGroup, TestSiLUGrad,
                        testing::Combine(EltwiseGradDynShapeTestCases, SiLUGradOpTypeCases));
}  // namespace ops
}  // namespace mindspore
