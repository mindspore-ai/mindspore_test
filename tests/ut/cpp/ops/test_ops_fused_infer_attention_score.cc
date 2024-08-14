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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name.h"
#include "infer/ops_func_impl/fused_infer_attention_score.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct FusedInferAttentionScoreParams {
  ShapeVector qkv_shape;
  TypePtr qkv_dtype;
  ValuePtr input_layout_value;
  ValuePtr head_num_value;
  ShapeVector out1_shape;
  TypePtr out1_type;
  ShapeVector out2_shape;
  TypePtr out2_type;
};

class TestFusedInferAttentionScore : public TestOps,
                                     public testing::WithParamInterface<FusedInferAttentionScoreParams> {};

TEST_P(TestFusedInferAttentionScore, dyn_shape) {
  const auto &param = GetParam();
  auto fused_infer_attention_score_func_impl = std::make_shared<FusedInferAttentionScoreFuncImpl>();
  auto prim = std::make_shared<Primitive>("FusedInferAttentionScore");
  auto none = std::make_shared<abstract::AbstractNone>();

  auto qkv = std::make_shared<abstract::AbstractTensor>(param.qkv_dtype, param.qkv_shape);
  ASSERT_NE(qkv, nullptr);
  abstract::AbstractBasePtr input_layout = nullptr;
  if (param.input_layout_value == nullptr) {
    input_layout = none;
  } else {
    input_layout = param.input_layout_value->ToAbstract();
  }
  abstract::AbstractBasePtr head_num = nullptr;
  if (param.head_num_value == nullptr) {
    head_num = none;
  } else {
    head_num = param.head_num_value->ToAbstract();
  }
  MS_EXCEPTION_IF_NULL(input_layout);
  MS_EXCEPTION_IF_NULL(head_num);

  auto expect_out1_shape = std::make_shared<abstract::Shape>(param.out1_shape);
  auto expect_out2_shape = std::make_shared<abstract::Shape>(param.out2_shape);
  ASSERT_NE(expect_out1_shape, nullptr);
  ASSERT_NE(expect_out2_shape, nullptr);
  auto expect_shape =
    std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({expect_out1_shape, expect_out2_shape}));
  auto expect_out1_dtype = std::make_shared<TensorType>(param.out1_type);
  auto expect_out2_dtype = std::make_shared<TensorType>(param.out2_type);
  ASSERT_NE(expect_out1_dtype, nullptr);
  ASSERT_NE(expect_out2_dtype, nullptr);
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{expect_out1_dtype, expect_out2_dtype});

  // execute
  auto input_none = std::make_shared<abstract::AbstractNone>();
  auto input_scalar = std::make_shared<abstract::AbstractScalar>();
  std::vector<AbstractBasePtr> input_args = {
    qkv,          qkv,          qkv,          input_none,   input_none,   input_none,   input_none,
    input_none,   input_none,   input_none,   input_none,   input_none,   input_none,   input_none,
    input_none,   input_none,   input_none,   head_num,     input_scalar, input_scalar, input_scalar,
    input_layout, input_scalar, input_scalar, input_scalar, input_scalar, input_scalar, input_scalar};
  auto out_shape = fused_infer_attention_score_func_impl->InferShape(prim, input_args);
  auto out_dtype = fused_infer_attention_score_func_impl->InferType(prim, input_args);
  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

INSTANTIATE_TEST_CASE_P(TestFusedInferAttentionScoreGroup, TestFusedInferAttentionScore,
                        testing::Values(FusedInferAttentionScoreParams{{16, 5, 4, 128},
                                                                       kFloat16,
                                                                       CreateScalar<int64_t>(1),  // BNSD
                                                                       CreateScalar<int64_t>(5),
                                                                       {16, 5, 4, 128},
                                                                       kFloat16,
                                                                       {16, 5, 4, 1},
                                                                       kFloat32},
                                        FusedInferAttentionScoreParams{{-1, 5, -1, -1},
                                                                       kBFloat16,
                                                                       CreateScalar<int64_t>(1),  // BNSD
                                                                       CreateScalar<int64_t>(5),
                                                                       {-1, 5, -1, -1},
                                                                       kBFloat16,
                                                                       {-1, 5, -1, 1},
                                                                       kFloat32},
                                        FusedInferAttentionScoreParams{{-1, -1, 5, -1},
                                                                       kInt8,
                                                                       CreateScalar<int64_t>(3),  // BSND
                                                                       CreateScalar<int64_t>(5),
                                                                       {-1, -1, 5, -1},
                                                                       kInt8,
                                                                       {-1, 5, -1, 1},
                                                                       kFloat32},
                                        FusedInferAttentionScoreParams{{-1, -1, -1},
                                                                       kFloat16,
                                                                       CreateScalar<int64_t>(0),  // BSH
                                                                       CreateScalar<int64_t>(5),
                                                                       {-1, -1, -1},
                                                                       kFloat16,
                                                                       {-1, 5, -1, 1},
                                                                       kFloat32}));

}  // namespace ops
}  // namespace mindspore
