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

#include "test_view.h"
#include "mindspore/ops/view/narrow_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewNarrow : public TestView {
 public:
  TestViewNarrow() {}
};

/// Feature: Narrow strides calculator
/// Description: Test view Narrow strides calculator is right
/// Expectation: success
TEST_F(TestViewNarrow, View) {
  auto prim = std::make_shared<Primitive>("Narrow");
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({1, 2, 4});
  int64_t input_dim = 2;
  int64_t input_start = 1;
  int64_t input_length = 2;
  auto dim_ = MakeValue(input_dim);
  auto start_ = MakeValue(input_start);
  auto length_ = MakeValue(input_length);
  std::vector<ValuePtr> inputs_a;
  inputs_a.emplace_back(input_tensor);
  inputs_a.emplace_back(dim_);
  inputs_a.emplace_back(start_);
  inputs_a.emplace_back(length_);
  auto storage_info = NarrowCalc(prim, inputs_a);
  std::vector<int64_t> expect_shape({1, 2, 2});
  std::vector<int64_t> expect_strides({8, 4, 1});
  size_t expect_offset = 1;
  ASSERT_FALSE(storage_info.empty());
  ASSERT_FALSE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->shape == expect_shape);
  ASSERT_TRUE(storage_info[0]->strides == expect_strides);
  ASSERT_TRUE(storage_info[0]->storage_offset == expect_offset);
}
}  // namespace ops
}  // namespace mindspore
