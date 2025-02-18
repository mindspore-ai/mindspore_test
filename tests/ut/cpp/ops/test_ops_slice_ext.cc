/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "ops/utils/general_infer_utils.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct SliceExtInferValueParams {
  tensor::TensorPtr input;
  ValuePtr dim;
  ValuePtr start;
  ValuePtr end;
  ValuePtr step;
  tensor::TensorPtr out;
};

class TestSliceExtInferValue : public TestOps, public testing::WithParamInterface<SliceExtInferValueParams> {};

TEST_P(TestSliceExtInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.dim, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);

  ASSERT_NE(param.start, nullptr);
  auto start = param.start->ToAbstract();
  ASSERT_NE(start, nullptr);

  ASSERT_NE(param.end, nullptr);
  auto end = param.end->ToAbstract();
  ASSERT_NE(end, nullptr);

  ASSERT_NE(param.step, nullptr);
  auto step = param.step->ToAbstract();
  ASSERT_NE(step, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, dim, start, end, step};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimSliceExt, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "SliceExt have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "SliceExt can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestSliceExtInferValue, TestSliceExtInferValue,
  testing::Values(
    SliceExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                             CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), CreateScalar<int64_t>(1),
                             CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6})},
    SliceExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                             CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1),
                             CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 3}, std::vector<float>{1, 2, 3})},
    SliceExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                             CreateScalar<int64_t>(0), CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), CreateScalar<int64_t>(2),
                             CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 3}, std::vector<float>{1, 2, 3})},
    SliceExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                             CreateScalar<int64_t>(1), CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), CreateScalar<int64_t>(1),
                             CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, 2, 4, 5})},
    SliceExtInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}),
                             CreateScalar<int64_t>(2), CreateScalar<int64_t>(0), CreateScalar<int64_t>(-1), CreateScalar<int64_t>(1),
                             CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 2}, std::vector<float>{1, 2, 4, 5})}));
}  // namespace ops
}  // namespace mindspore
