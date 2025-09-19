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

#include <mindspore/core/include/ir/core_ops_primitive.h>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/convert_bfloat16.h"
#include "utils/anf_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore::graphkernel::test {
class TestConvertBFloat16 : public GraphKernelCommonTestSuite {};

namespace {
void CheckType(const AnfNodePtr &node, const std::vector<TypeId> &target_inputs_type, TypeId target_output_type) {
  // check abstract
  auto abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto tensor = abstract->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(tensor->element());
  auto type_ptr = tensor->element()->GetType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->type_id();
  if (type != target_output_type) {
    MS_LOG(ERROR) << "abstract expect data type: " << TypeIdToString(target_output_type);
    MS_LOG(ERROR) << "abstract output data type: " << TypeIdToString(type);
    ASSERT_TRUE(false);
  }
  // check build info
  MS_EXCEPTION_IF_NULL(node->kernel_info());
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  MS_EXCEPTION_IF_NULL(build_info);
  auto inputs_type = build_info->GetAllInputDeviceTypes();
  for (size_t i = 0; i < target_inputs_type.size(); ++i) {
    if (inputs_type[i] != target_inputs_type[i]) {
      MS_LOG(ERROR) << "build_info expect input[" << i << "] data type: " << TypeIdToString(target_inputs_type[i]);
      MS_LOG(ERROR) << "build_info output input[" << i << "] data type: " << TypeIdToString(inputs_type[i]);
      ASSERT_TRUE(false);
    }
  }
  auto outputs_type = build_info->GetAllOutputDeviceTypes();
  ASSERT_TRUE(!outputs_type.empty());
  if (outputs_type[0] != target_output_type) {
    MS_LOG(ERROR) << "build_info expect data type: " << TypeIdToString(target_output_type);
    MS_LOG(ERROR) << "build_info output data type: " << TypeIdToString(outputs_type[0]);
    ASSERT_TRUE(false);
  }
}

/**
 *  sub_graph(p0: bf16, p1: bf16, p2: bf16) {
 *    %0(bf16) = Add(p0, p1)
 *    %1(bool) = Less(%0, p2)
 *    return %1
 *  }
 *  ---------->
 *  sub_graph(p0: bf16, p1: bf16, p2: bf16) {
 *    %0(fp32) = Cast(p0, fp32)
 *    %1(fp32) = Cast(p1, fp32)
 *    %2(fp32) = Add(%0, %1)
 *    %3(fp32) = Cast(p2, fp32)
 *    %3(bool) = Less(%2, %3)
 *    return %3
 *  }
 */
void Run1(GraphKernelCommonTestSuite *t) {
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kBFloat16, {32, 32});
  auto x1 = c.NewTensorInput("x1", kBFloat16, {32, 32});
  auto x2 = c.NewTensorInput("x2", kBFloat16, {32, 32});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {x0, x1}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Less", {op1, x2}, {});
  c.SetOutput(op2);
  auto fg = c.GetGraph();
  t->RunPass(fg,
             {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::ConvertBFloat16>()});

  bool check1 = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      auto sub_nodes = TopoSort(sub_graph->get_return());
      for (auto sub_node : sub_nodes) {
        if (sub_node == nullptr) {
          continue;
        }
        if (IsPrimitiveCNode(sub_node, prim::kPrimAdd)) {
          CheckType(sub_node, {kNumberTypeFloat32, kNumberTypeFloat32}, kNumberTypeFloat32);
        } else if (IsPrimitiveCNode(sub_node, prim::kPrimLess)) {
          CheckType(sub_node, {kNumberTypeFloat32, kNumberTypeFloat32}, kNumberTypeBool);
          check1 = true;
        }
      }
    }
  }
  ASSERT_TRUE(check1);
}

/**
 *  sub_graph(p0: bf16, p1: bf16, p2: bf16) {
 *    %0(bf16) = Add(p0, p1)
 *    %1(bf16) = Abs(p2)
 *    %2(bf16) = MatMul(%0, %1)
 *    return %2
 *  }
 *  ---------->
 *  sub_graph(p0: bf16, p1: bf16, p2: bf16) {
 *    %0(fp32) = Cast(p0, fp32)
 *    %1(fp32) = Cast(p1, fp32)
 *    %2(fp32) = Add(%0, %1)
 *    %3(fp32) = Cast(p2, fp32)
 *    %4(fp32) = Abs(%3)
 *    %5(bf16) = Cast(%2, bf16)
 *    %6(bf16) = Cast(%4, bf16)
 *    %7(bf16) = MatMul(%5, %6)
 *    return %7
 *  }
 */
void Run2(GraphKernelCommonTestSuite *t) {
  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = "--enable_cluster_ops=MatMul";
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kBFloat16, {1024, 1024});
  auto x1 = c.NewTensorInput("x1", kBFloat16, {1024, 1024});
  auto x2 = c.NewTensorInput("x2", kBFloat16, {1024, 1024});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {x0, x1}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Abs", {x2}, {});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto op3 = c.NewCNodeWithBuildInfo("MatMul", {op1, op2, trans_a, trans_b}, {});
  c.SetOutput(op3);
  auto fg = c.GetGraph();
  t->RunPass(fg,
             {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::ConvertBFloat16>()});

  bool check2 = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      auto sub_nodes = TopoSort(sub_graph->get_return());
      for (auto sub_node : sub_nodes) {
        if (sub_node == nullptr) {
          continue;
        }
        if (IsPrimitiveCNode(sub_node, prim::kPrimAdd)) {
          CheckType(sub_node, {kNumberTypeFloat32, kNumberTypeFloat32}, kNumberTypeFloat32);
        } else if (IsPrimitiveCNode(sub_node, prim::kPrimAbs)) {
          CheckType(sub_node, {kNumberTypeFloat32}, kNumberTypeFloat32);
        } else if (IsPrimitiveCNode(sub_node, prim::kPrimMatMul)) {
          CheckType(sub_node, {kNumberTypeBFloat16, kNumberTypeBFloat16}, kNumberTypeBFloat16);
          check2 = true;
        }
      }
    }
  }
  ASSERT_TRUE(check2);
}

/**
 *  sub_graph(p0: bf16, p1: bf16) {
 *    %0(bf16) = Add(p0, p1)
 *    %1(fp32) = Cast(%0, fp32)
 *    return %0, %1
 *  }
 *  ---------->
 *  sub_graph(p0: bf16, p1: bf16) {
 *    %0(fp32) = Cast(p0, fp32)
 *    %1(fp32) = Cast(p1, fp32)
 *    %2(fp32) = Add(%0, %1)
 *    %3(bf16) = Cast(%2, bf16)
 *    %4(fp32) = Cast(%3, fp32)
 *    return %3, %4
 *  }
 */
void Run3(GraphKernelCommonTestSuite *t) {
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kBFloat16, {32, 32});
  auto x1 = c.NewTensorInput("x1", kBFloat16, {32, 32});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {x0, x1}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Cast", {op1, c.NewValueNode<int64_t>(kNumberTypeFloat32)}, {});
  auto op3 = c.NewCNodeWithBuildInfo("MakeTuple", {op1, op2}, {});
  c.SetOutput(op3);
  auto fg = c.GetGraph();
  t->RunPass(fg,
             {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::ConvertBFloat16>()});

  bool check3 = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      auto sub_nodes = TopoSort(sub_graph->get_return());
      for (auto sub_node : sub_nodes) {
        if (sub_node == nullptr) {
          continue;
        }
        if (IsPrimitiveCNode(sub_node, prim::kPrimAdd)) {
          CheckType(sub_node, {kNumberTypeFloat32, kNumberTypeFloat32}, kNumberTypeFloat32);
        } else if (IsPrimitiveCNode(sub_node, prim::kPrimMakeTuple)) {
          auto cnode = sub_node->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(cnode);
          auto input1 = cnode->input(1);
          auto input2 = cnode->input(2);
          auto input2_cnode = input2->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(input2_cnode);
          if (input2_cnode->input(1) != input1) {
            MS_LOG(ERROR) << "input1: " << input1->DebugString();
            MS_LOG(ERROR) << "input2: " << input2->DebugString();
            ASSERT_TRUE(false);
          }
          check3 = true;
        }
      }
    }
  }
  ASSERT_TRUE(check3);
}

/**
 *  sub_graph(p0: bf16) {
 *    %0(bf16) = Reshape(p0)
 *    %1(bf16) = Abs(%0)
 *    %2(bf16) = Reshape(%1)
 *    return %0
 *  }
 *  ---------->
 *  sub_graph(p0: bf16) {
 *    %0(bf16) = Reshape(p0)
 *    %1(fp32) = Cast(%0, fp32)
 *    %2(fp32) = Abs(%1)
 *    %3(fp32) = Reshape(%2)
 *    %4(bf16) = Cast(%3, bf16)
 *    return %4
 *  }
 */
void Run4(GraphKernelCommonTestSuite *t) {
  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = "--enable_cluster_ops=Reshape";
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kBFloat16, {2, 6});
  auto op1 = c.NewCNodeWithBuildInfo("Reshape", {x0, c.NewValueNode<std::vector<int64_t>>({12})}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Abs", {op1}, {});
  auto op3 = c.NewCNodeWithBuildInfo("Reshape", {op2, c.NewValueNode<std::vector<int64_t>>({3, 4})}, {});
  c.SetOutput(op3);
  auto fg = c.GetGraph();
  t->RunPass(fg,
             {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::ConvertBFloat16>()});

  bool check4 = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = GetCNodeFuncGraph(node);
      auto sub_nodes = TopoSort(sub_graph->get_return());
      for (auto sub_node : sub_nodes) {
        if (sub_node == nullptr) {
          continue;
        }
        if (IsPrimitiveCNode(sub_node, prim::kPrimReshape)) {
          auto cnode = sub_node->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(cnode);
          auto input = cnode->input(1);
          MS_EXCEPTION_IF_NULL(input);
          if (!input->isa<CNode>()) {
            // The first Reshape keeps bf16
            CheckType(sub_node, {kNumberTypeBFloat16}, kNumberTypeBFloat16);
          } else {
            // The second Reshape uses fp32
            CheckType(sub_node, {kNumberTypeFloat32}, kNumberTypeFloat32);
          }
          check4 = true;
        }
      }
    }
  }
  ASSERT_TRUE(check4);
}
}  // namespace

/// Feature: Test graph kernel ConvertBFloat16 pass
/// Description: Run ConvertBFloat16 with different fuse ops
/// Expectation: The Cast op inserted by ConvertBFloat16 should match with expect
TEST_F(TestConvertBFloat16, convert_bfloat16) {
  SetDeviceTarget(kAscendDevice);
  Run1(this);
  Run2(this);
  Run3(this);
  Run4(this);
}
}  // namespace mindspore::graphkernel::test