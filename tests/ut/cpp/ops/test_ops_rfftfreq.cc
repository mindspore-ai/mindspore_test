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
#include "infer/ops_func_impl/rfftfreq.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "op_def/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
#define F32(x) (static_cast<float>((x)))
struct RFFTFreqShape {
  ValuePtr n;
  ValuePtr d;
  ShapeVector out_shape;
};

struct RFFTFreqType {
  ValuePtr dtype;
  TypePtr out_type;
};

class TestRFFTFreq : public TestOps, public testing::WithParamInterface<std::tuple<RFFTFreqShape, RFFTFreqType>> {};

TEST_P(TestRFFTFreq, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  RFFTFreqFuncImpl rfftfreq_func_impl;
  auto primitive = std::make_shared<Primitive>("RFFTFreq");
  ASSERT_NE(primitive, nullptr);
  auto n = shape_param.n->ToAbstract();
  auto d = shape_param.d->ToAbstract();
  auto dtype = type_param.dtype->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {n, d, dtype};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = rfftfreq_func_impl.InferShape(primitive, input_args);
  auto out_dtype = rfftfreq_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto rfftfreq_shape_cases = testing::Values(RFFTFreqShape{CreateScalar(I64(4)), CreateScalar(F32(1.0)), {3}},
                                            RFFTFreqShape{CreateScalar(I64(7)), CreateScalar(F32(2.5)), {4}},
                                            RFFTFreqShape{CreateScalar(I64(9)), CreateScalar(F32(3.7)), {5}},
                                            RFFTFreqShape{CreateScalar(I64(1)), CreateScalar(F32(4.2)), {1}});

auto rfftfreq_type_cases = testing::ValuesIn({RFFTFreqType{CreateScalar<int64_t>(kNumberTypeBFloat16), kBFloat16},
                                              RFFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat16), kFloat16},
                                              RFFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat32), kFloat32},
                                              RFFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat64), kFloat64},
                                              RFFTFreqType{CreateScalar<int64_t>(kNumberTypeComplex64), kComplex64},
                                              RFFTFreqType{CreateScalar<int64_t>(kNumberTypeComplex128), kComplex128}});

INSTANTIATE_TEST_CASE_P(TestRFFTFreqGroup, TestRFFTFreq, testing::Combine(rfftfreq_shape_cases, rfftfreq_type_cases));
}  // namespace ops
}  // namespace mindspore
