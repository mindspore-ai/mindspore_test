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

#include <string>

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
class TestCheckOutput : public TestSymbolEngine {};

/// Feature: check output dtype for symbolic value
/// Description: TupleToTensor, input and output dtypes are int64
/// Expectation: infer success
TEST_F(TestCheckOutput, tupletotensor_case1_int64) {
  mindspore::test::ConstructGraph cg;
  auto p1 = cg.NewTupleInput("p1", {{kInt64, {}}, {kInt64, {}}}, true);
  auto node = cg.NewCNodeWithBuildInfo("TupleToTensor", {p1, cg.NewValueNode(MakeValue<int64_t>(kNumberTypeInt64))});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicValue(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
}

/// Feature: check output dtype for symbolic value
/// Description: TupleToTensor, input and output dtypes are float32
/// Expectation: infer success
TEST_F(TestCheckOutput, tupletotensor_case2_float32) {
  mindspore::test::ConstructGraph cg;
  auto p1 = cg.NewTupleInput("p1", {{kFloat32, {}}, {kFloat32, {}}}, true);
  auto node = cg.NewCNodeWithBuildInfo("TupleToTensor", {p1, cg.NewValueNode(MakeValue<int64_t>(kNumberTypeFloat32))});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicValue(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
}

/// Feature: check output dtype for symbolic value
/// Description: TupleToTensor, input dtype is int64, output dtype is float32
/// Expectation: infer failed. if supported, this case can be dropped.
TEST_F(TestCheckOutput, tupletotensor_case3_int64_fp32) {
  mindspore::test::ConstructGraph cg;
  auto p1 = cg.NewTupleInput("p1", {{kInt64, {}}, {kInt64, {}}}, true);
  auto node = cg.NewCNodeWithBuildInfo("TupleToTensor", {p1, cg.NewValueNode(MakeValue<int64_t>(kNumberTypeFloat32))});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicValue(node);
  SaveIR(cg.GetGraph());
  ASSERT_FALSE(helper_->SupportInfer());
}

/// Feature: check output dtype for symbolic value
/// Description: ScalarToTensor, input and output dtypes are int64
/// Expectation: infer success
TEST_F(TestCheckOutput, scalartotensor_case1_int64) {
  mindspore::test::ConstructGraph cg;
  auto p1 = cg.NewScalarInput("p1", kInt64);
  auto node = cg.NewCNodeWithBuildInfo("ScalarToTensor", {p1, cg.NewValueNode(MakeValue<int64_t>(kNumberTypeInt64))});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicValue(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
}

/// Feature: check output dtype for symbolic value
/// Description: ScalarToTensor, input dtype is int64, output dtype is float32
/// Expectation: infer success
TEST_F(TestCheckOutput, scalartotensor_case2_int64_fp32) {
  mindspore::test::ConstructGraph cg;
  auto p1 = cg.NewScalarInput("p1", kInt64);
  auto node = cg.NewCNodeWithBuildInfo("ScalarToTensor", {p1, cg.NewValueNode(MakeValue<int64_t>(kNumberTypeFloat32))});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicValue(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
}
}  // namespace mindspore::symshape::test
