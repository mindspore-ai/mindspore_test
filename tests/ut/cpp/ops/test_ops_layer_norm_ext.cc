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
#include "infer/ops_func_impl/layer_norm_ext.h"
#include "common/common_test.h"
#include "ir/primitive.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct LayerNormExtOpParams {
  ShapeVector input_x_shape;
  TypePtr input_x_type;
  ValuePtr normalized_shape;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  ShapeVector beta_shape;
  TypePtr beta_type;
  float epsilon;

  ShapeVector output_x_shape;
  TypePtr output_x_type;
  ShapeVector mean_shape;
  ShapeVector rstd_shape;
};

class TestLayerNormExt : public TestOps, public testing::WithParamInterface<LayerNormExtOpParams> {};

TEST_P(TestLayerNormExt, layer_norm_ext_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("LayerNormExt");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.input_x_type, param.input_x_shape);
  ASSERT_NE(input_x, nullptr);
  auto normalized_shape = param.normalized_shape->ToAbstract();
  ASSERT_NE(normalized_shape, nullptr);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  ASSERT_NE(gamma, nullptr);
  auto beta = std::make_shared<abstract::AbstractTensor>(param.beta_type, param.beta_shape);
  ASSERT_NE(beta, nullptr);
  auto epsilon = std::make_shared<abstract::AbstractScalar>(param.epsilon);
  ASSERT_NE(epsilon, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(input_x), std::move(normalized_shape), std::move(gamma),
                                                    std::move(beta), std::move(epsilon)};
  auto infer_impl = std::make_shared<LayerNormExtFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shapes_ptr = infer_impl->InferShape(primitive, input_args);
  std::shared_ptr<abstract::TupleShape> infer_shapes =
    std::dynamic_pointer_cast<abstract::TupleShape>(infer_shapes_ptr);
  ASSERT_NE(infer_shapes, nullptr);
  auto infer_types_ptr = infer_impl->InferType(primitive, input_args);
  std::shared_ptr<Tuple> infer_types = std::dynamic_pointer_cast<Tuple>(infer_types_ptr);
  ASSERT_NE(infer_types, nullptr);
  auto expect_output_x_shape = std::make_shared<abstract::TensorShape>(param.output_x_shape);
  ASSERT_NE(expect_output_x_shape, nullptr);
  auto expect_output_x_type = std::make_shared<TensorType>(param.output_x_type);
  ASSERT_NE(expect_output_x_type, nullptr);
  auto expect_mean_shape = std::make_shared<abstract::TensorShape>(param.mean_shape);
  ASSERT_NE(expect_mean_shape, nullptr);
  auto expect_rstd_shape = std::make_shared<abstract::TensorShape>(param.rstd_shape);
  ASSERT_NE(expect_rstd_shape, nullptr);
  ASSERT_TRUE(*((*infer_shapes)[0]) == *expect_output_x_shape);
  ASSERT_TRUE(*((*infer_shapes)[1]) == *expect_mean_shape);
  ASSERT_TRUE(*((*infer_shapes)[2]) == *expect_rstd_shape);
  ASSERT_TRUE(*((*infer_types)[0]) == *expect_output_x_type);
}

INSTANTIATE_TEST_CASE_P(TestLayerNormExtGroup, TestLayerNormExt,
                        testing::Values(LayerNormExtOpParams{{-2},
                                                             kFloat32,
                                                             CreateScalar(kValueAny),
                                                             {-2},
                                                             kFloat32,
                                                             {-2},
                                                             kFloat32,
                                                             0.5,
                                                             {-2},
                                                             kFloat32,
                                                             {-2},
                                                             {-2}},
                                        LayerNormExtOpParams{{2, 3, 4},
                                                             kFloat32,
                                                             CreateTuple({I64(3), I64(4)}),
                                                             {3, 4},
                                                             kFloat32,
                                                             {3, 4},
                                                             kFloat32,
                                                             0.5,
                                                             {2, 3, 4},
                                                             kFloat32,
                                                             {2, 1, 1},
                                                             {2, 1, 1}}));
}  // namespace ops
}  // namespace mindspore
