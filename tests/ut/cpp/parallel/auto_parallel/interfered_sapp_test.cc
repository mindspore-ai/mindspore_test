/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"

namespace mindspore {
namespace parallel {
class TestInterferedSapp : public UT::Common {
 public:
  TestInterferedSapp() {}

  void SetUp() {}
  void TearDown() {}
};

/// Feature: test TransposeCombine.
/// Description:
/// Expectation: success
TEST_F(TestInterferedSapp, test_transpose_combine) {
  std::vector<int64_t> transpose_mapping1 = {3, 2, 1, 0};
  std::vector<int64_t> transpose_mapping2 = {2, 3, 0, 1};

  std::vector<int64_t> updated = TransposeCombine(transpose_mapping1, transpose_mapping2);
  std::vector<int64_t> expected = {1, 0, 3, 2};
  ASSERT_EQ(updated.size(), expected.size()) << "Vectors updated and expected are of unequal length";

  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(updated[i], expected[i]) << "Vectors updated and expected differ at index " << i;
  }
}

/// Feature: test error check of TransposeCombine.
/// Description:
/// Expectation: Throw Exception
TEST_F(TestInterferedSapp, test_transpose_combine_check) {
  std::vector<int64_t> transpose_mapping1 = {1, 2, 3, 0};
  std::vector<int64_t> transpose_mapping2 = {2};
  EXPECT_ANY_THROW(TransposeCombine(transpose_mapping1, transpose_mapping2));
}

/// Feature: test ReshapeCombine.
/// Description:
/// Expectation: success
TEST_F(TestInterferedSapp, test_reshape_combine) {
  std::vector<int64_t> reshape1_v1 = {2};
  std::vector<int64_t> reshape1_v2 = {2};
  std::vector<int64_t> reshape1_v3 = {3};
  std::vector<int64_t> reshape1_v4 = {3};
  std::vector<int64_t> reshape2_v1;
  std::vector<int64_t> reshape2_v2;
  std::vector<int64_t> reshape2_v3 = {0, 1};
  std::vector<int64_t> reshape2_v4 = {2, 3};
  std::vector<int64_t> reshape3_v1 = {0};
  std::vector<int64_t> reshape3_v2 = {1};
  std::vector<int64_t> reshape3_v3 = {2};
  std::vector<int64_t> reshape3_v4 = {3};
  std::vector<std::vector<int64_t>> reshape_mapping1;
  reshape_mapping1.push_back(reshape1_v1);
  reshape_mapping1.push_back(reshape1_v2);
  reshape_mapping1.push_back(reshape1_v3);
  reshape_mapping1.push_back(reshape1_v4);
  std::vector<std::vector<int64_t>> reshape_mapping2;
  reshape_mapping2.push_back(reshape2_v1);
  reshape_mapping2.push_back(reshape2_v2);
  reshape_mapping2.push_back(reshape2_v3);
  reshape_mapping2.push_back(reshape2_v4);
  std::vector<std::vector<int64_t>> reshape_mapping3;
  reshape_mapping3.push_back(reshape3_v1);
  reshape_mapping3.push_back(reshape3_v2);
  reshape_mapping3.push_back(reshape3_v3);
  reshape_mapping3.push_back(reshape3_v4);
  std::vector<std::vector<int64_t>> updated = ReshapeCombine(reshape_mapping1, reshape_mapping2);
  ASSERT_EQ(updated.size(), reshape_mapping3.size());
  for (size_t i = 0; i < updated.size(); i++) {
    for (size_t j = 0; j < updated[i].size(); j++) {
      ASSERT_EQ(updated[i][j], reshape_mapping3[i][j]);
    }
  }
}

/// Feature: test error check of ReshapeCombine.
/// Description:
/// Expectation: Throw Exception
TEST_F(TestInterferedSapp, test_reshape_combine_check) {
  std::vector<int64_t> reshape1_v1 = {2};
  std::vector<int64_t> reshape1_v2 = {2};
  std::vector<int64_t> reshape1_v3 = {3};
  std::vector<int64_t> reshape1_v4 = {3};
  std::vector<int64_t> reshape2_v1 = {0, 1};
  std::vector<int64_t> reshape2_v2 = {2, 3};
  std::vector<std::vector<int64_t>> reshape_mapping1;
  reshape_mapping1.push_back(reshape1_v1);
  reshape_mapping1.push_back(reshape1_v2);
  reshape_mapping1.push_back(reshape1_v3);
  reshape_mapping1.push_back(reshape1_v4);
  std::vector<std::vector<int64_t>> reshape_mapping2;
  reshape_mapping2.push_back(reshape2_v1);
  reshape_mapping2.push_back(reshape2_v2);
  EXPECT_ANY_THROW(ReshapeCombine(reshape_mapping1, reshape_mapping2));
}
}  // namespace parallel
}  // namespace mindspore
