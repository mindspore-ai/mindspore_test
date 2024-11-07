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
#include "infer/ops_func_impl/meshgrid.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
struct MeshgridParams {
  bool dynamic_len;
  ShapeArray x_shapes;
  TypePtr x_type;
  ValuePtr indexing_value;
  ShapeArray out_shape;
  TypePtr out_type;
};

class TestMeshgrid : public TestOps, public testing::WithParamInterface<MeshgridParams> {};

TEST_P(TestMeshgrid, dyn_shape) {
  const auto &param = GetParam();

  AbstractBasePtrList inputs;
  inputs.reserve(param.x_shapes.size());
  for (auto x_shape : param.x_shapes) {
    auto input = std::make_shared<abstract::AbstractTensor>(param.x_type, x_shape);
    ASSERT_NE(input, nullptr);
    inputs.push_back(input);
  }

  auto tuple_x = std::make_shared<abstract::AbstractTuple>(inputs);
  ASSERT_NE(tuple_x, nullptr);
  if (param.dynamic_len) {
    tuple_x->CheckAndConvertToDynamicLenSequence();
  }

  auto indexing = std::make_shared<abstract::AbstractScalar>(param.indexing_value, kInt64);
  ASSERT_NE(indexing, nullptr);

  std::vector<abstract::BaseShapePtr> out_shape;
  for (auto it : param.out_shape) {
    out_shape.push_back(std::make_shared<abstract::TensorShape>(it));
  }

  auto expect_shape = std::make_shared<abstract::TupleShape>(out_shape);
  auto expect_type =
    std::make_shared<Tuple>(std::vector<TypePtr>(out_shape.size(), std::make_shared<TensorType>(param.out_type)));
  DoFuncImplInferAndCompare<MeshgridFuncImpl>("Meshgrid", abstract::AbstractBasePtrList{tuple_x, indexing},
                                              expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestMeshgridGroup, TestMeshgrid,
  testing::Values(MeshgridParams{false, {{3}, {4}}, kFloat32, CreatePyInt(1), {{4, 3}, {4, 3}}, kFloat32},
                  MeshgridParams{false, {{1}, {3}}, kFloat64, CreatePyInt(0), {{1, 3}, {1, 3}}, kFloat64},
                  MeshgridParams{false, {{-1}, {4}}, kUInt8, CreatePyInt(1), {{4, -1}, {4, -1}}, kUInt8},
                  MeshgridParams{false, {{-1}, {-1}}, kUInt16, CreatePyInt(1), {{-1, -1}, {-1, -1}}, kUInt16},
                  MeshgridParams{true, {{3}, {5}}, kUInt64, CreatePyInt(1), {{5, 3}, {5, 3}}, kUInt64},
                  MeshgridParams{true, {{4}, {6}}, kInt32, CreatePyInt(0), {{4, 6}, {4, 6}}, kInt32},
                  MeshgridParams{true, {{-1}, {4}}, kBFloat16, CreatePyInt(1), {{4, -1}, {4, -1}}, kBFloat16}));
}  // namespace mindspore::ops
