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
#include <memory>
#include "common/common_test.h"
#include "infer/ops_func_impl/inner_comm_reduce_scatter.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ReduceScatterShapeParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr rank_size;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestReduceScatter : public TestOps, public testing::WithParamInterface<ReduceScatterShapeParams> {};

TEST_P(TestReduceScatter, dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  InnerCommReduceScatterFuncImpl all_gather_func_impl;
  auto prim = std::make_shared<Primitive>("InnerCommReduceScatter");

  auto out_dtype = all_gather_func_impl.InferType(prim, {input_x, param.rank_size->ToAbstract()});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = all_gather_func_impl.InferShape(prim, {input_x, param.rank_size->ToAbstract()});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestReduceScatter, TestReduceScatter,
  testing::Values(ReduceScatterShapeParams{{6, 4, 5}, kFloat32, CreateScalar<int64_t>(2), {3, 4, 5}, kFloat32},
                  ReduceScatterShapeParams{{24, 4, 5}, kFloat16, CreateScalar<int64_t>(8), {3, 4, 5}, kFloat16},
                  ReduceScatterShapeParams{{-1, 4, 5}, kInt64, CreateScalar<int64_t>(2), {-1, 4, 5}, kInt64},
                  ReduceScatterShapeParams{{8, -1, -1}, kInt64, CreateScalar<int64_t>(2), {4, -1, -1}, kInt64},
                  ReduceScatterShapeParams{{-2}, kInt64, CreateScalar<int64_t>(2), {-2}, kInt64}));
}  // namespace ops
}  // namespace mindspore
