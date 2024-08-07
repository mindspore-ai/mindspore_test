/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/histc_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct HistcExtShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ValuePtr bins;
  ValuePtr min;
  ValuePtr max;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestHistcExt : public TestOps, public testing::WithParamInterface<HistcExtShapeParams> {};

TEST_P(TestHistcExt, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto bins = param.bins->ToAbstract();
  auto min = param.min->ToAbstract();
  auto max = param.max->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);

  HistcExtFuncImpl histc_ext_func_impl;
  auto prim = std::make_shared<Primitive>("HistcExt");

  auto out_shape = histc_ext_func_impl.InferShape(prim, {input, bins, min, max});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
  auto out_dtype = histc_ext_func_impl.InferType(prim, {input, bins, min, max});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
}

OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(HistcExt, EltwiseOpParams);
OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(
  HistcExt,
  testing::Values(
    EltwiseOpParams{{2, 3}, kInt32, {4}, kInt32, {CreateScalar<int64_t>(4), CreateScalar(0.0), CreateScalar(3.0)}},
    EltwiseOpParams{{2, 3}, kFloat16, {4}, kFloat16, {CreateScalar<int64_t>(4), CreateScalar(0.0), CreateScalar(3.0)}},
    EltwiseOpParams{
      {2, 3}, kFloat32, {4}, kFloat32, {CreateScalar<int64_t>(4), CreateScalar(0.0), CreateScalar(3.0)}}));

INSTANTIATE_TEST_CASE_P(
  TestHistcExt, TestHistcExt,
  testing::Values(
    HistcExtShapeParams{{3}, kInt32, CreateScalar<int64_t>(4), CreateScalar(0.0), CreateScalar(3.0), {4}, kInt32},
    HistcExtShapeParams{{3}, kInt32, CreateScalar(kValueAny), CreateScalar(0.0), CreateScalar(3.0), {-1}, kInt32},
    HistcExtShapeParams{{3, 4}, kInt32, CreateScalar<int64_t>(4), CreateScalar(0.0), CreateScalar(3.0), {4}, kInt32}));
}  // namespace ops
}  // namespace mindspore
