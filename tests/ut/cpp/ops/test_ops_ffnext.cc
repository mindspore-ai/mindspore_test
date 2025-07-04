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
#include "infer/ops_func_impl/ffn_ext.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace ops {
struct TestFFNExtParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestFFNExt : public TestOps, public testing::WithParamInterface<TestFFNExtParams> {};

TEST_P(TestFFNExt, scatter_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<FFNExtFuncImpl>(kNameFFNExt, input_args, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestFFNExtGroup, TestFFNExt,
                        testing::Values(TestFFNExtParams{{5, 5120}, kFloat16, {5, 5120}, kFloat16},
                                        TestFFNExtParams{{5, 5120}, kInt8, {5, 5120}, kInt8},
                                        TestFFNExtParams{{5, 5120}, kBFloat16, {5, 5120}, kBFloat16}));
}  // namespace ops
}  // namespace mindspore
