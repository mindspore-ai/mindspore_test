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
#include "infer/ops_func_impl/rfft.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct RFFTShape {
  ShapeVector x_shape;
  ValuePtr n;
  ValuePtr dim;
  ValuePtr norm;
  ShapeVector out_shape;
};

struct RFFTType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestRFFT : public TestOps, public testing::WithParamInterface<std::tuple<RFFTShape, RFFTType>> {};

TEST_P(TestRFFT, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  RFFTFuncImpl rfft_func_impl;
  auto primitive = std::make_shared<Primitive>("RFFT");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto n = shape_param.n->ToAbstract();
  auto dim = shape_param.dim->ToAbstract();
  auto norm = shape_param.norm->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, n, dim, norm};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = rfft_func_impl.InferShape(primitive, input_args);
  auto out_dtype = rfft_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto rfft_shape_cases =
  testing::Values(RFFTShape{{4, 4}, CreateScalar(I64(4)), CreateScalar(I64(-1)), CreateScalar(I64(0)), {4, 3}},
                  RFFTShape{{4, 4}, CreateScalar(I64(2)), CreateScalar(I64(-1)), CreateScalar(I64(0)), {4, 2}},
                  RFFTShape{{4, 4}, CreateScalar(I64(8)), CreateScalar(I64(-1)), CreateScalar(I64(1)), {4, 5}},
                  RFFTShape{{4, 4}, CreateScalar(I64(2)), CreateScalar(I64(0)), CreateScalar(I64(0)), {2, 4}},
                  RFFTShape{{4, 4}, CreateScalar(I64(8)), CreateScalar(I64(0)), CreateScalar(I64(1)), {5, 4}});

auto rfft_type_cases = testing::ValuesIn({
  RFFTType{kInt16, kComplex64},
  RFFTType{kInt32, kComplex64},
  RFFTType{kInt64, kComplex64},
  RFFTType{kFloat16, kComplex64},
  RFFTType{kFloat32, kComplex64},
  RFFTType{kFloat64, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestRFFTGroup, TestRFFT, testing::Combine(rfft_shape_cases, rfft_type_cases));
}  // namespace ops
}  // namespace mindspore
