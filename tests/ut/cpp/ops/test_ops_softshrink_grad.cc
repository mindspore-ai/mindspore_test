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
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "infer/ops_func_impl/softshrink_grad.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct SoftshrinkGradShape {
  ShapeVector gradients_shape;
  ShapeVector features_shape;
  ValuePtr lambd;
  ShapeVector out_shape;
};

struct SoftshrinkGradDtype {
  TypePtr gradients_type;
  TypePtr features_type;
  TypePtr out_type;
};

class TestSoftshrinkGrad : public TestOps,
                        public testing::WithParamInterface<std::tuple<SoftshrinkGradShape, SoftshrinkGradDtype>> {};

TEST_P(TestSoftshrinkGrad, softshrink_grad_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  SoftShrinkGradFuncImpl softshrink_grad_func_impl;
  auto prim = std::make_shared<Primitive>("SoftShrinkGrad");

  auto gradients = std::make_shared<abstract::AbstractTensor>(dtype_param.gradients_type, shape_param.gradients_shape);
  auto features = std::make_shared<abstract::AbstractTensor>(dtype_param.features_type, shape_param.features_shape);

  ASSERT_NE(gradients, nullptr);
  ASSERT_NE(features, nullptr);
  auto lambd = shape_param.lambd->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = softshrink_grad_func_impl.InferShape(prim, {gradients, features, lambd});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = softshrink_grad_func_impl.InferType(prim, {gradients, features, lambd});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto SoftshrinkGradOpShapeTestCases = testing::ValuesIn({
  /* static */
  SoftshrinkGradShape{{2, 3, 4}, {2, 3, 4}, CreateScalar(0.5), {2, 3, 4}},
  SoftshrinkGradShape{{1, 2, 3}, {1, 2, 3}, CreateScalar(False), {1, 2, 3}},
  SoftshrinkGradShape{{3, 4, 2}, {3, 4, 2}, CreateScalar(4), {3, 4, 2}},
  /* dynamic shape */
  SoftshrinkGradShape{{-1}, {-1}, CreateScalar(0.3), {-1}},
  SoftshrinkGradShape{{-1, 2, 4}, {-1, 2, 4}, CreateScalar(True), {-1, 2, 4}},
  SoftshrinkGradShape{{5, 3, -1, 2, 1}, {5, 3, -1, 2, 1}, CreateScalar(3), {5, 3, -1, 2, 1}},
  SoftshrinkGradShape{{5, 3, -1, 2}, {5, 3, -1, 2, 1, 4, 7, 4}, CreateScalar(-0.4), {5, 3, -1, 2, 1, 4, 7, 4}},
  /* dynamic rank */
  SoftshrinkGradShape{{-2}, {-2}, CreateScalar(0.5), {-2}},
  SoftshrinkGradShape{{-2}, {-2}, CreateScalar(False), {-2}},
  SoftshrinkGradShape{{-2}, {-2}, CreateScalar(4), {-2}},
});

auto SoftshrinkGradOpTypeTestCases = testing::ValuesIn({
  SoftshrinkGradDtype{kFloat16, kFloat16, kFloat16},
  SoftshrinkGradDtype{kFloat32, kFloat32, kFloat32},
  SoftshrinkGradDtype{kBFloat16, kBFloat16, kBFloat16},
});

OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(SoftShrinkGrad, MultiInputOpParams);
OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(
  SoftShrinkGrad,
  testing::Values(MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                  MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
                  MultiInputOpParams{{{2, 3}, {2, 3}}, {kBFloat16, kBFloat16}, {{2, 3}}, {kBFloat16}, {}}));
INSTANTIATE_TEST_CASE_P(TestSoftshrinkGrad, TestSoftshrinkGrad,
                        testing::Combine(SoftshrinkGradOpShapeTestCases, SoftshrinkGradOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
