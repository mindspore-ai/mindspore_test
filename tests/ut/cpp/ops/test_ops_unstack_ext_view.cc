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
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "infer/ops_func_impl/unstack_ext_view.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct UnstackExtOpParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ValuePtr dim;
  ShapeArray out_shapes;
  TypePtr out_type;
};
class TestUnstackExt : public TestOps, public testing::WithParamInterface<UnstackExtOpParams> {};

TEST_P(TestUnstackExt, unstack_ext_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("UnstackExtView");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(input, nullptr);

  AbstractBasePtrList output_abs;
  for (auto output_shape : param.out_shapes) {
      auto key_element = std::make_shared<abstract::AbstractTensor>(param.out_type, output_shape);
      output_abs.push_back(key_element);
  }
  auto expect_abs = std::make_shared<abstract::AbstractTuple>(output_abs);
  ASSERT_NE(expect_abs, nullptr);
  auto dim_value = GetValue<int64_t>(param.dim);
  ASSERT_LT(dim_value, SizeToLong(param.input_shape.size()));
  if (dim_value < 0) {
    dim_value += SizeToLong(param.input_shape.size());
  }
  if ((param.input_shape.size() == 1 && param.input_shape[0] == -2) ||
       param.input_shape[dim_value] == -1) {
      expect_abs->CheckAndConvertToDynamicLenSequence();
  }
  auto expect_shape = expect_abs->GetShape();
  auto expect_type = expect_abs->GetType();
  // infer
  auto infer_impl = GetOpFrontendFuncImplPtr("UnstackExtView");
  ASSERT_NE(infer_impl, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(input), dim};
  auto infer_shape_type = infer_impl->InferAbstract(primitive, input_args);
  ASSERT_NE(infer_shape_type, nullptr);
  auto infer_shape = infer_shape_type->GetShape();
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_shape_type->GetType();
  ASSERT_NE(infer_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestUnstackExt, TestUnstackExt,
  testing::Values(UnstackExtOpParams{{2, 3, 4}, kFloat32, CreateScalar<int64_t>(0), {{3, 4}, {3, 4}}, kFloat32},
                  UnstackExtOpParams{{2, 3, 4}, kInt64, CreateScalar<int64_t>(-2), {{2, 4}, {2, 4}, {2, 4}}, kInt64},
                  UnstackExtOpParams{{2, 3}, kComplex64, CreateScalar<int64_t>(1), {{2}, {2}, {2}}, kComplex64},
                  UnstackExtOpParams{{2, -1}, kFloat16, CreateScalar<int64_t>(0), {{-1}, {-1}}, kFloat16},
                  UnstackExtOpParams{{-1, -1}, kFloat32, CreateScalar<int64_t>(-1), {{-1}}, kFloat32},
                  UnstackExtOpParams{{-2}, kComplex128, CreateScalar<int64_t>(0), {{-2}}, kComplex128}));
}  // namespace ops
}  // namespace mindspore
