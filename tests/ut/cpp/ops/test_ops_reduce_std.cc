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
#include "infer/ops_func_impl/reduce_std.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
struct ReduceStdParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ValuePtr unbiased;
  ValuePtr keep_dims;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestReduceStd : public TestOps, public testing::WithParamInterface<ReduceStdParams> {};

TEST_P(TestReduceStd, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);
  auto keep_dims = param.keep_dims->ToAbstract();
  ASSERT_NE(keep_dims, nullptr);
  auto unbiased = param.unbiased->ToAbstract();
  ASSERT_NE(unbiased, nullptr);

  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>(2, std::make_shared<abstract::Shape>(param.out_shape)));
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>(2, std::make_shared<TensorType>(param.out_type)));
  DoFuncImplInferAndCompare<ReduceStdFuncImpl>(kNameReduceStd, {x, axis, unbiased, keep_dims}, expect_shape,
                                               expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestReduceStdGroup, TestReduceStd,
  testing::Values(
    ReduceStdParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(true), {2, 1, 4}, kFloat32},
    ReduceStdParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), CreateScalar(false), {2, 4}, kFloat32},
    ReduceStdParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({0, 1}), CreateScalar(true), CreateScalar(true), {1, 1, 4}, kFloat32},
    ReduceStdParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({0, 1}), CreateScalar(false), CreateScalar(false), {4}, kFloat32},
    ReduceStdParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(true),
                    CreateScalar(true),
                    {-1, 1, -1},
                    kFloat32},
    ReduceStdParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({kValueAny, 1}), CreateScalar(false), CreateScalar(false), {-1}, kFloat32},
    ReduceStdParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(true),
                    CreateScalar(true),
                    {-1, -1, -1},
                    kFloat32},
    ReduceStdParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(false),
                    CreateScalar(false),
                    {-1},
                    kFloat32},
    ReduceStdParams{{2, 3, 4}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {-1, -1, -1}, kFloat32},
    ReduceStdParams{{2, 3, 4}, kFloat32, kValueAny, CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceStdParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32},
    ReduceStdParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({1}), kValueAny, kValueAny, {-2}, kFloat32},
    ReduceStdParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({1, 2}), kValueAny, kValueAny, {-2}, kFloat32},
    ReduceStdParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(true), {-1, 1, 4}, kFloat32},
    ReduceStdParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), CreateScalar(false), {-1, 4}, kFloat32},
    ReduceStdParams{
      {-1, 3, 4}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(true), CreateScalar(true), {1, 3, 1}, kFloat32},
    ReduceStdParams{
      {-1, 3, 4}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(false), CreateScalar(false), {3}, kFloat32},
    ReduceStdParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(true),
                    CreateScalar(true),
                    {-1, 1, -1},
                    kFloat32},
    ReduceStdParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(false),
                    CreateScalar(false),
                    {-1},
                    kFloat32},
    ReduceStdParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(true),
                    CreateScalar(true),
                    {-1, -1, -1},
                    kFloat32},
    ReduceStdParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(false),
                    CreateScalar(false),
                    {-1},
                    kFloat32},
    ReduceStdParams{{-1, -1, 4}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {-1, -1, -1}, kFloat32},
    ReduceStdParams{{-1, -1, 4}, kFloat32, kValueAny, CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceStdParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32},
    ReduceStdParams{{-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), kValueAny, kValueAny, {-2}, kFloat32},
    ReduceStdParams{{-1, -1, 4}, kFloat32, CreatePyIntTuple({1, 2}), kValueAny, kValueAny, {-2}, kFloat32},
    ReduceStdParams{{-2}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(true), {-2}, kFloat32},
    ReduceStdParams{{-2}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceStdParams{
      {-2}, kFloat32, CreatePyIntTuple({kValueAny, 1}), CreateScalar(true), CreateScalar(true), {-2}, kFloat32},
    ReduceStdParams{{-2}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(true), {-2}, kFloat32},
    ReduceStdParams{{-2}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
