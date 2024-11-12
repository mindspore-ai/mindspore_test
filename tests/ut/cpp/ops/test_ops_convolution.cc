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
#include "infer/ops_func_impl/convolution.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
struct ConvolutionParams {
  ShapeVector input_shape;
  ShapeVector weight_shape;
  ShapeVector bias_shape;
  ValuePtr stride;       // tuple[int]
  ValuePtr padding;      // tuple[int]
  ValuePtr dilation;       // tuple[int]
  ValuePtr transposed;       // bool
  ValuePtr outputPadding;       // tuple[int]
  ValuePtr groups;       // tuple[int]
  ShapeVector out_shape;
};

class TestConvolution : public TestOps, public testing::WithParamInterface<ConvolutionParams> {};

TEST_P(TestConvolution, dyn_shape) {
  const auto &param = GetParam();

  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, param.input_shape);
  ASSERT_NE(input_abs, nullptr);
  auto weight_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, param.weight_shape);
  ASSERT_NE(weight_abs, nullptr);
  auto bias_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, param.bias_shape);
  ASSERT_NE(bias_abs, nullptr);

  auto stride_abs = param.stride->ToAbstract();
  ASSERT_NE(stride_abs, nullptr);
  auto padding_abs = param.padding->ToAbstract();
  ASSERT_NE(padding_abs, nullptr);
  auto dilation_abs = param.dilation->ToAbstract();
  ASSERT_NE(dilation_abs, nullptr);
  auto transposed_abs = param.transposed->ToAbstract();
  ASSERT_NE(transposed_abs, nullptr);
  auto outputPadding_abs = param.outputPadding->ToAbstract();
  ASSERT_NE(outputPadding_abs, nullptr);
  auto groups_abs = param.groups->ToAbstract();
  ASSERT_NE(stride_abs, nullptr);

  auto count_include_pad = CreateScalar<bool>(false);
  auto count_include_pad_abs = count_include_pad->ToAbstract();
  auto divisor_override = CreateScalar<int64_t>(int64_t(1));
  auto divisor_override_abs = divisor_override->ToAbstract();

  auto prim = std::make_shared<Primitive>("Convolution");
  auto infer_impl = std::make_shared<ConvolutionFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);

  // for abstract based infer
  std::vector<AbstractBasePtr> input_args{input_abs, weight_abs, bias_abs, stride_abs, padding_abs, dilation_abs,
                                          transposed_abs, outputPadding_abs, groups_abs};
  auto inferred_shape = infer_impl->InferShape(prim, input_args);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ShapeCompare(inferred_shape, expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestConvolution, TestConvolution,
  testing::Values(ConvolutionParams{ShapeVector{-2}, ShapeVector{-2}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{-2}},

                  ConvolutionParams{ShapeVector{-2}, ShapeVector{2, 2, 1, 1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{-1, 2, -1, -1}},

                  ConvolutionParams{ShapeVector{3, 2, 3, 3}, ShapeVector{-2}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{3, -1, -1, -1}},

                  ConvolutionParams{ShapeVector{3, 2, 3, 3}, ShapeVector{-1,-1,-1,-1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{3, -1, -1, -1}},

                  ConvolutionParams{ShapeVector{-1, -1, 3, 3}, ShapeVector{2, 2, 1, 1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{-1, 2, 3, 3}},

                  ConvolutionParams{ShapeVector{-1, 2, 3, 3}, ShapeVector{2, 2, 1, 1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{-1, 2, 3, 3}},

                  ConvolutionParams{ShapeVector{2, 3, 8, 8}, ShapeVector{3, 3, 2, 2}, ShapeVector{3}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({1, 1}), kValueAny, CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{2, 3, -1, -1}},

                  ConvolutionParams{ShapeVector{2, 3, 8, 8}, ShapeVector{3, 3, 2, 2}, ShapeVector{-2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({1, 1}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{2, 3, 9, 9}},

                  ConvolutionParams{ShapeVector{2, 3, 8, 8}, ShapeVector{3, 3, 2, 2}, ShapeVector{3}, kValueAny,
                                    CreatePyIntTuple({1, 1}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{2, 3, -1, -1}},
                  
                  ConvolutionParams{ShapeVector{3, 2, 3, 3}, ShapeVector{2, 2, -1, -1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    kValueAny, CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{3, 2, -1, -1}},
                  
                  ConvolutionParams{ShapeVector{3, 2, 3, 3}, ShapeVector{2, 2, 1, 1}, ShapeVector{2}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({0, 0}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{3, 2, 3, 3}},
                  
                  ConvolutionParams{ShapeVector{2, 3, 8, 8}, ShapeVector{3, 3, 2, 2}, ShapeVector{3}, CreatePyIntTuple({1, 1}),
                                    CreatePyIntTuple({1, 1}), CreatePyIntTuple({1, 1}), CreateScalar<bool>(false),
                                    CreatePyIntTuple({1, 1}), CreatePyInt(1), ShapeVector{2, 3, 9, 9}}));
}  // namespace mindspore::ops
