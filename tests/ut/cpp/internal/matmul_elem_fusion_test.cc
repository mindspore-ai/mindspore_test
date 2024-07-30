/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/graph_optimizer_test_framework.h"
#include "ops/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/matmul_elemwise_fusion.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "utils/phase.h"

namespace mindspore {
namespace opt {
class MatmulElemFusionUT : public UT::Common {
 public:
  MatmulElemFusionUT() {}
};

/// Feature: A backend pass: MatmulElemAddFusion
/// Description: Convert MatMul + Add to FusedMatMulElemBinary
/// Expectation: After optimize, match FusedMatMulElemBinary.
TEST_F(MatmulElemFusionUT, MatmulElemAddFusionTest) {
  std::map<std::string, std::string> jit_config;
  jit_config["infer_boost"] = "on";
  PhaseManager::GetInstance().set_jit_config(jit_config);
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1024, 1024});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1024, 1024});
  auto input_2 = c.NewTensorInput("input_2", kFloat16, {1024});
  auto ta = c.NewValueNode(MakeValue<bool>(false));
  auto tb = c.NewValueNode(MakeValue<bool>(false));
  auto matmul = c.NewCNode("MatMul", {input_0, input_1, ta, tb}, {});
  auto add = c.NewCNode("Add", {matmul, input_2}, {});

  c.SetOutput(add);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulElemAddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("input_2").AddCNode(
    "matmul_add", {std::make_shared<Primitive>("FusedMatMulElemBinary"), "input_0", "input_1", "input_2"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: MatmulElemGeluFusion
/// Description: Convert MatMul + Add to FusedMatMulElemUnary
/// Expectation: After optimize, match FusedMatMulElemUnary.
TEST_F(MatmulElemFusionUT, MatmulElemGeluFusionTest) {
  std::map<std::string, std::string> jit_config;
  jit_config["infer_boost"] = "on";
  PhaseManager::GetInstance().set_jit_config(jit_config);
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1024, 1024});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1024, 1024});
  auto ta = c.NewValueNode(MakeValue<bool>(false));
  auto tb = c.NewValueNode(MakeValue<bool>(false));
  auto matmul = c.NewCNode("MatMul", {input_0, input_1, ta, tb}, {});
  auto gelu = c.NewCNode("GeLU", {matmul}, {});

  c.SetOutput(gelu);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulElemGeluFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddCNode(
    "matmul_gelu", {std::make_shared<Primitive>("FusedMatMulElemUnary"), "input_0", "input_1"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: MatmulElemReluFusion
/// Description: Convert MatMul + Add to FusedMatMulElemUnary
/// Expectation: After optimize, match FusedMatMulElemUnary.
TEST_F(MatmulElemFusionUT, MatmulElemReluFusionTest) {
  std::map<std::string, std::string> jit_config;
  jit_config["infer_boost"] = "on";
  PhaseManager::GetInstance().set_jit_config(jit_config);
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1024, 1024});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1024, 1024});
  auto ta = c.NewValueNode(MakeValue<bool>(false));
  auto tb = c.NewValueNode(MakeValue<bool>(false));
  auto matmul = c.NewCNode("MatMul", {input_0, input_1, ta, tb}, {});
  auto relu = c.NewCNode("ReLU", {matmul}, {});

  c.SetOutput(relu);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::MatmulElemReluFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddCNode(
    "matmul_relu", {std::make_shared<Primitive>("FusedMatMulElemUnary"), "input_0", "input_1"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore
