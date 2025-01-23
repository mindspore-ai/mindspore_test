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
#include <memory>
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name.h"
#include "infer/ops_func_impl/transpose_ext.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ir/dtype/tensor_type.h"
#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
#define I64(x) (static_cast<int64_t>((x)))

struct TransExtParams {
  bool dynamic_len;
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr dim0;
  ValuePtr dim1;
  ShapeVector out_shape;
};

class TestTransposeExt : public TestOps, public testing::WithParamInterface<TransExtParams> {};

TEST_P(TestTransposeExt, dyn_shape) {
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  auto dim0_abs = param.dim0->ToAbstract();
  auto dim1_abs = param.dim1->ToAbstract();
  ASSERT_NE(dim0_abs, nullptr);
  ASSERT_NE(dim1_abs, nullptr);

  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);
  DoFuncImplInferAndCompare<TransposeExtFuncImpl>(kNameTransposeExt, {x, dim0_abs, dim1_abs}, expect_shape,
                                                  expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestTransposeExt, TestTransposeExt,
  testing::Values(
    TransExtParams{false, {2, 3, 4, 5}, kFloat32, CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), {3, 2, 4, 5}},
    TransExtParams{false, {2, 3, -1, 5}, kFloat32, CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), {-1, 3, 2, 5}},
    TransExtParams{true, {-2}, kFloat32, CreateScalar(kValueAny), CreateScalar(kValueAny), {-2}},
    TransExtParams{false, {2, -1, 4, -1}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(3), {2, -1, -1, 4}},
    TransExtParams{false, {2, 3, 4, 5}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(3), {2, 5, 4, 3}},
    TransExtParams{false, {2, 3, -1, 5}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(3), {2, 5, -1, 3}}));

struct TransposeExtInferValueParams {
  tensor::TensorPtr input;
  ValuePtr dim0;
  ValuePtr dim1;
  tensor::TensorPtr out;
};

class TestTransposeExtInferValue : public TestOps, public testing::WithParamInterface<TransposeExtInferValueParams> {};

TEST_P(TestTransposeExtInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.dim0, nullptr);
  auto dim0 = param.dim0->ToAbstract();
  ASSERT_NE(dim0, nullptr);

  ASSERT_NE(param.dim1, nullptr);
  auto dim1 = param.dim1->ToAbstract();
  ASSERT_NE(dim1, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, dim0, dim1};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimTransposeExt, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "TransposeExt have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "TransposeExt can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestTransposeExtInferValue, TestTransposeExtInferValue,
  testing::Values(
    TransposeExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, 2, 3, 4}),
                                 CreateScalar<int64_t>(0), CreateScalar<int64_t>(1),
                                 CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, 3, 2, 4})},
    TransposeExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                                 CreateScalar<int64_t>(0), CreateScalar<int64_t>(1),
                                 CreateTensor<float>(kNumberTypeFloat32, ShapeVector{3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6})},
    TransposeExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                                 CreateScalar<int64_t>(0), CreateScalar<int64_t>(2),
                                 CreateTensor<float>(kNumberTypeFloat32, ShapeVector{3, 2, 1}, std::vector<float>{1, 4, 2, 5, 3, 6})},
    TransposeExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                                 CreateScalar<int64_t>(1), CreateScalar<int64_t>(2),
                                 CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6})}));
}  // namespace mindspore::ops
