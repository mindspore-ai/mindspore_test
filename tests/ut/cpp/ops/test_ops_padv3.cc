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
#include "infer/pad_v3.h"
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
struct PadV3Params {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector paddings_shape;
  TypePtr paddings_type;
  ShapeVector value_shape;
  TypePtr value_type;
  std::string mode;
  bool paddings_contiguous;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestPadV3 : public TestOps, public testing::WithParamInterface<PadV3Params> {};

TEST_P(TestPadV3, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto paddings = std::make_shared<abstract::AbstractTensor>(param.paddings_type, param.paddings_shape);
  auto value = std::make_shared<abstract::AbstractTensor>(param.value_type, param.value_shape);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(paddings, nullptr);
  ASSERT_NE(value, nullptr);

  auto prim = std::make_shared<Primitive>(kNamePadV3);
  prim->set_attr("mode", MakeValue<std::string>(param.mode));
  prim->set_attr("paddings_contiguous", MakeValue<bool>(param.paddings_contiguous));

  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  abstract::AbstractBasePtr out_abstract;
  if (param.mode == "constant") {
    out_abstract = opt::CppInferShapeAndType(prim, {input, paddings, value});
  } else {
    out_abstract = opt::CppInferShapeAndType(prim, {input, paddings});
  }
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestPadV3Group, TestPadV3,
  testing::Values(
    PadV3Params{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "constant", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "constant", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{-2}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "constant", true, {-2}, kFloat32},
    PadV3Params{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "reflect", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "reflect", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{-2}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "reflect", true, {-2}, kFloat32},
    PadV3Params{{1, -1, -1, -1}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "replicate", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{1, 2, 3, 4}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "replicate", true, {-1, -1, -1, -1}, kFloat32},
    PadV3Params{{-2}, kFloat32, {-1, -1}, kInt32, {}, kInt32, "replicate", true, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
