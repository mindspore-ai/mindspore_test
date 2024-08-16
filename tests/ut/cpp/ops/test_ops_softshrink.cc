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
#include "infer/ops_func_impl/softshrink.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct SoftShrinkShape {
  ShapeVector input_x_shape;
  ValuePtr lambd;
  ShapeVector out_shape;
};

struct SoftShrinkDtype {
  TypePtr input_x_type;
  TypePtr out_type;
};

class TestSoftshrink : public TestOps, public testing::WithParamInterface<std::tuple<SoftShrinkShape, SoftShrinkDtype>> {};

TEST_P(TestSoftshrink, softshrink_input_xn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  SoftShrinkFuncImpl softshrink_func_impl;
  auto prim = std::make_shared<Primitive>("Softshrink");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_x_type, shape_param.input_x_shape);
  auto lambd = shape_param.lambd->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = softshrink_func_impl.InferShape(prim, {input_x, lambd});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = softshrink_func_impl.InferType(prim, {input_x, lambd});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto SoftshrinkOpShapeTestCases = testing::ValuesIn({
  /* static */
  SoftShrinkShape{{2, 3, 4}, CreateScalar(0.5), {2, 3, 4}},
  SoftShrinkShape{{3, 4, 2}, CreateScalar(False), {3, 4, 2}},
  SoftShrinkShape{{4, 3, 2}, CreateScalar(2), {4, 3, 2}},
  /* dynamic shape */
  SoftShrinkShape{{-1}, CreateScalar(0.3), {-1}},
  SoftShrinkShape{{-1, 2, 4}, CreateScalar(False), {-1, 2, 4}},
  SoftShrinkShape{{5, 3, -1, 2, 1}, CreateScalar(4), {5, 3, -1, 2, 1}},
  SoftShrinkShape{{5, 3, -1, 2, 1, 4, 7, 4}, CreateScalar(-0.4), {5, 3, -1, 2, 1, 4, 7, 4}},
  /* dynamic rank */
  SoftShrinkShape{{-2}, CreateScalar(0.5), {-2}},
  SoftShrinkShape{{-2}, CreateScalar(True), {-2}},
  SoftShrinkShape{{-2}, CreateScalar(4), {-2}},
});

auto SoftshrinkOpTypeTestCases = testing::ValuesIn({
  SoftShrinkDtype{kFloat16, kFloat16},
  SoftShrinkDtype{kFloat32, kFloat32},
  SoftShrinkDtype{kBFloat16, kBFloat16},
});

OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(SoftShrink, EltwiseOpParams);
OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(SoftShrink,
                                    testing::Values(EltwiseOpParams{{2, 3}, kFloat16, {2, 3}, kFloat16, {}},
                                                    EltwiseOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {}},
                                                    EltwiseOpParams{{2, 3}, kBFloat16, {2, 3}, kBFloat16, {}}));

INSTANTIATE_TEST_CASE_P(TestSoftshrink, TestSoftshrink, testing::Combine(SoftshrinkOpShapeTestCases, SoftshrinkOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
