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
#include <memory>
#include "common/common_test.h"
#include "common/graph_optimizer_test_framework.h"
#include "op_def/sequence_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "include/transform/graph_ir/utils.h"
#define private public
#define protected public
#include "backend/common/pass/graph_view_replace_pass.h"
#undef private public
#undef protected public

namespace mindspore {

class TestConcatSplitTransposeView : public UT::Common {
 public:
  TestConcatSplitTransposeView() = default;
  ~TestConcatSplitTransposeView() override = default;
  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test graph view replace pass
/// Description: test correct situation
/// Expectation: concat ops will be replace by concatView
TEST_F(TestConcatSplitTransposeView, test_concat_with_correct_inputsize) {
  common::SetEnv("MS_DEV_VIEW_OP", "Concat");
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {16, 8});
  auto x2 = c.NewTensorInput("x2", kFloat32, {8, 16});
  auto axis = c.NewValueNode(MakeValue<int64_t>(0));
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul1 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto matmul2 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto concat1 = c.NewCNode("Concat", {matmul1, matmul2, axis});
  auto x3 = c.NewTensorInput("x3", kFloat32, {32, 16});
  auto add1 = c.NewCNode("Add", {concat1, x3});
  c.SetOutput(add1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("axis")
    .AddVar("x3")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("matmul2", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_b"})
    .AddCNode("concatView", {std::make_shared<Primitive>("ConcatView"), "matmul1", "matmul2", "axis"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "concatView", "x3"});
  EXPECT_TRUE(checker.build_pattern_map(g->output()));
  common::SetEnv("MS_DEV_VIEW_OP", "");
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: concat ops will not be replace by concatView
TEST_F(TestConcatSplitTransposeView, test_concat_with_wrong_inputsize) {
  common::SetEnv("MS_DEV_VIEW_OP", "Concat");
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat32, {3, 2});
  auto axis = c.NewValueNode(MakeValue<int64_t>(0));
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul1 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto matmul2 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto concat1 = c.NewCNode("Concat", {matmul1, matmul2, axis});
  auto x3 = c.NewTensorInput("x3", kFloat32, {4, 2});
  auto add1 = c.NewCNode("Add", {concat1, x3});
  c.SetOutput(add1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("axis")
    .AddVar("x3")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("matmul2", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_b"})
    .AddCNode("concatView", {std::make_shared<Primitive>("ConcatView"), "matmul1", "matmul2", "axis"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "concatView", "x3"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
  common::SetEnv("MS_DEV_VIEW_OP", "");
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: concat ops will not be replace by concatView
TEST_F(TestConcatSplitTransposeView, test_concat_with_output_bountry) {
  common::SetEnv("MS_DEV_VIEW_OP", "Split");
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat32, {3, 2});
  auto axis = c.NewValueNode(MakeValue<int64_t>(0));
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul1 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto matmul2 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto concat1 = c.NewCNode("Concat", {matmul1, matmul2, axis});
  c.SetOutput(concat1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("axis")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("matmul2", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_b"})
    .AddCNode("concatView", {std::make_shared<Primitive>("ConcatView"), "matmul1", "matmul2", "axis"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
  common::SetEnv("MS_DEV_VIEW_OP", "");
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: concat ops will not be replace by concatView
TEST_F(TestConcatSplitTransposeView, test_concat_with_wrong_input) {
  common::SetEnv("MS_DEV_VIEW_OP", "Concat,Split");
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat32, {3, 2});
  auto axis = c.NewValueNode(MakeValue<int64_t>(0));
  auto x3 = c.NewTensorInput("x3", kFloat32, {2, 2});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul1 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto concat1 = c.NewCNode("Concat", {matmul1, x3, axis});
  auto x4 = c.NewTensorInput("x4", kFloat32, {4, 2});
  auto add1 = c.NewCNode("Add", {concat1, x4});
  c.SetOutput(add1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("axis")
    .AddVar("x3")
    .AddVar("x4")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("concatView", {std::make_shared<Primitive>("ConcatView"), "matmul1", "x3", "axis"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "concatView", "x4"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
  common::SetEnv("MS_DEV_VIEW_OP", "");
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: concat ops will not be replace by concatView
TEST_F(TestConcatSplitTransposeView, test_concat_with_wrong_output) {
  common::SetEnv("MS_DEV_VIEW_OP", "Concat,Split");
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto x2 = c.NewTensorInput("x2", kFloat32, {3, 2});
  auto axis = c.NewValueNode(MakeValue<int64_t>(0));
  auto shape0 = c.NewValueNode(MakeValue(std::vector<int64_t>{2, 4}));
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto matmul1 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto matmul2 = c.NewCNode("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto concat1 = c.NewCNode("Concat", {matmul1, matmul2, axis});
  auto reshape1 = c.NewCNode("Reshape", {concat1, shape0});
  c.SetOutput(reshape1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("axis")
    .AddVar("shape0")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("matmul2", {std::make_shared<Primitive>("MatMul"), "x1", "x2", "trans_a", "trans_a"})
    .AddCNode("concatView", {std::make_shared<Primitive>("ConcatView"), "matmul1", "matmul2", "axis"})
    .AddCNode("reshape", {std::make_shared<Primitive>("Reshape"), "concatView", "shape0"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
  common::SetEnv("MS_DEV_VIEW_OP", "");
}

/// Feature: test graph view replace pass
/// Description: test correct situation
/// Expectation: transpose ops will be replace by TransposeView
TEST_F(TestConcatSplitTransposeView, test_transpose_with_good_output) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 3});
  auto shape0 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 0}));
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto transpose1 = c.NewCNode("Transpose", {x1, shape0});
  auto matmul1 = c.NewCNode("MatMul", {x1, transpose1, trans_a, trans_b}, {});
  c.SetOutput(matmul1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("shape0")
    .AddVar("trans_a")
    .AddVar("trans_b")
    .AddCNode("transposeview", {std::make_shared<Primitive>("TransposeView"), "x1", "shape0"})
    .AddCNode("matmul1", {std::make_shared<Primitive>("MatMul"), "x1", "transposeview", "trans_a", "trans_b"});
  EXPECT_TRUE(checker.build_pattern_map(g->output()));
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: transpose ops will not  be replace by TransposeView
TEST_F(TestConcatSplitTransposeView, test_transpose_with_wrong_output) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 2});
  auto x2 = c.NewTensorInput("x1", kFloat32, {2, 2});
  auto shape0 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 0}));
  auto transpose1 = c.NewCNode("Transpose", {x1, shape0});
  auto add1 = c.NewCNode("Add", {transpose1, x2});
  c.SetOutput(add1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("shape0")
    .AddCNode("transposeview", {std::make_shared<Primitive>("TransposeView"), "x1", "shape0"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "transposeview", "x2"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
}

/// Feature: test graph view replace pass
/// Description: test wrong situation
/// Expectation: transpose ops will be replace by TransposeView
TEST_F(TestConcatSplitTransposeView, test_transpose_with_output_boundry) {
  test::ConstructGraph c;
  auto x1 = c.NewTensorInput("x1", kFloat32, {2, 2});
  auto shape0 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 0}));
  auto transpose1 = c.NewCNode("Transpose", {x1, shape0});
  c.SetOutput(transpose1);
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  test::RunPass(g, {std::make_shared<opt::GraphViewReplacePass>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1").AddVar("shape0").AddCNode(
    "transposeview", {std::make_shared<Primitive>("TransposeView"), "x1", "shape0"});
  EXPECT_FALSE(checker.build_pattern_map(g->output()));
}

}  // namespace mindspore