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
#include "infer/ops_func_impl/betainc.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
struct TestBetaincParams {
  ShapeVector a_shape;
  TypePtr a_type;
  ShapeVector b_shape;
  TypePtr b_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestBetainc : public TestOps, public testing::WithParamInterface<TestBetaincParams> {};

TEST_P(TestBetainc, betainc_dyn_shape) {
  const auto &param = GetParam();
  auto a = std::make_shared<abstract::AbstractTensor>(param.a_type, param.a_shape);
  auto b = std::make_shared<abstract::AbstractTensor>(param.b_type, param.b_shape);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(x, nullptr);
  auto infer_impl = std::make_shared<BetaincFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(a), std::move(b), std::move(x)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<BetaincFuncImpl>(kNameBetainc, input_args, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestBetainc, TestBetainc,
  testing::Values(TestBetaincParams{{-1, -1}, kFloat32, {-2}, kFloat32, {-2}, kFloat32, {-1, -1}, kFloat32},
                  TestBetaincParams{{2, 2}, kFloat32, {-1, -1}, kFloat32, {-2}, kFloat32, {2, 2}, kFloat32},
                  TestBetaincParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32, {-1, -1}, kFloat32, {-1, -1}, kFloat32},
                  TestBetaincParams{{-2}, kFloat32, {-2}, kFloat32, {-2}, kFloat32, {-2}, kFloat32},
                  TestBetaincParams{
                    {2, -1, -1}, kFloat32, {-1, 3, -1}, kFloat32, {-1, -1, 4}, kFloat32, {2, 3, 4}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
