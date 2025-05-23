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
#include <string>
#include "common/common_test.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace ops {
struct PadV3GradParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector paddings_shape;
  TypePtr paddings_type;
  Mode mode;
  bool paddings_contiguous;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestPadV3Grad : public TestOps, public testing::WithParamInterface<PadV3GradParams> {};

TEST_P(TestPadV3Grad, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto paddings = std::make_shared<abstract::AbstractTensor>(param.paddings_type, param.paddings_shape);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(paddings, nullptr);
  auto mode_value = std::make_shared<Int64Imm>(param.mode);
  auto mode = mode_value->ToAbstract();
  auto paddings_contiguous_value = std::make_shared<BoolImm>(param.paddings_contiguous);
  auto paddings_contiguous = paddings_contiguous_value->ToAbstract();
  ASSERT_NE(mode, nullptr);
  ASSERT_NE(paddings_contiguous, nullptr);

  auto prim = std::make_shared<Primitive>(kNamePadV3Grad);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto out_abstract = opt::CppInferShapeAndType(prim, {input, paddings, mode, paddings_contiguous});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestPadV3GradGroup, TestPadV3Grad,
  testing::Values(
    PadV3GradParams{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, Mode::CONSTANT, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, Mode::CONSTANT, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{-2}, kFloat32, {-1, -1}, kInt32, Mode::CONSTANT, true, {-2}, kFloat32},
    PadV3GradParams{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, Mode::REFLECT, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, Mode::REFLECT, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{-2}, kFloat32, {-1, -1}, kInt32, Mode::REFLECT, true, {-2}, kFloat32},
    PadV3GradParams{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, Mode::CIRCULAR, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, Mode::CIRCULAR, true, {-1, -1, -1, -1}, kFloat32},
    PadV3GradParams{{-2}, kFloat32, {-1, -1}, kInt32, Mode::CIRCULAR, true, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
