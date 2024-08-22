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
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/trunc.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "op_def/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace ops {

struct TruncShape {
  ShapeVector x_shape;
  ShapeVector out_shape;
};
struct TruncType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestTrunc : public TestOps, public testing::WithParamInterface<std::tuple<TruncShape, TruncType>> {};

TEST_P(TestTrunc, trunc_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  TruncFuncImpl trunc_func_impl;
  auto prim = std::make_shared<Primitive>("Trunc");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = trunc_func_impl.InferShape(prim, {x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = trunc_func_impl.InferType(prim, {x});
  ASSERT_TRUE(*out_dtype == *expect_dtype);

  // simple infer
  auto x_value = std::make_shared<tensor::Tensor>(dtype_param.x_type->type_id(), shape_param.x_shape);
  auto expect_shape_simple_infer = {shape_param.out_shape};
  auto expect_dtype_simple_infer = {dtype_param.out_type};
  DoFuncImplSimpleInferAndCompare<TruncFuncImpl>(kNameTrunc, {x_value}, {expect_shape_simple_infer},
                                                {expect_dtype_simple_infer});
}

auto TruncOpShapeTestCases = testing::ValuesIn({
  /* static */
  TruncShape{{2}, {2}},
  TruncShape{{2, 3, 4}, {2, 3, 4}},
  /* dynamic shape */
  TruncShape{{-1}, {-1}},
  TruncShape{{-1, 2, 4}, {-1, 2, 4}},
  TruncShape{{5, 3, -1, 2, 1}, {5, 3, -1, 2, 1}},
  /* dynamic rank */
  TruncShape{{-2}, {-2}},
});

auto TruncOpTypeTestCases = testing::ValuesIn({
  TruncType{kFloat16, kFloat16},
  TruncType{kBFloat16, kBFloat16},
  TruncType{kFloat32, kFloat32},
  TruncType{kInt8, kInt8},
  TruncType{kInt32, kInt32},
  TruncType{kUInt8, kUInt8}, 
});

INSTANTIATE_TEST_CASE_P(TestTrunc, TestTrunc, testing::Combine(TruncOpShapeTestCases, TruncOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
