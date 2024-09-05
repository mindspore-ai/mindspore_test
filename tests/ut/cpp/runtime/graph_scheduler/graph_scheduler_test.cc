/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "tests/ut/cpp/common/device_common_test.h"

#include "op_def/comparison_ops.h"
#include "op_def/framework_ops.h"
#include "op_def/math_ops.h"

namespace mindspore {
namespace runtime {
using namespace test;
class GraphSchedulerTest : public UT::Common {
 public:
  GraphSchedulerTest() {}
};

namespace {
FuncGraphPtr BuildFuncGraph() {
  std::vector<int64_t> shp{2, 2};
  auto func_graph = std::make_shared<FuncGraph>();
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_x = func_graph->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_y = func_graph->add_parameter();
  parameter_y->set_abstract(abstract_y);
  return func_graph;
}

/// Feature: unify runtime.
/// Description: build parameter.
/// Expectation: success.
FuncGraphPtr BuildFuncGraphWithParameter() {
  std::vector<int64_t> shp_4{4, 4};
  std::vector<int64_t> shp_8{2, 8};
  auto func_graph_shp4 = std::make_shared<FuncGraph>();
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_4);
  auto parameter_x = func_graph_shp4->add_parameter();
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_8);
  auto parameter_y = func_graph_shp4->add_parameter();
  parameter_y->set_abstract(abstract_y);
  return func_graph_shp4;
}

/// Feature: unify runtime.
/// Description: build singlecallfuncgraph.
/// Expectation: success.
FuncGraphPtr BuildSingleCallFuncGraph() {
  auto root_func_graph = BuildFuncGraphWithParameter();
  auto mul_func_graph = BuildFuncGraphWithParameter();
  std::vector<int64_t> shp_4{4, 4};

  // root graph
  auto parameters = root_func_graph->parameters();
  // add
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameters[0], parameters[0]};
  auto add = root_func_graph->NewCNode(add_inputs);
  auto add_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  add->set_abstract(add_abs);
  // call
  // scalar
  auto value_2 = MakeValue(2);
  auto valuenode_2 = NewValueNode(value_2);
  valuenode_2->set_abstract(value_2->ToAbstract());
  std::vector<AnfNodePtr> call_inputs{NewValueNode(mul_func_graph), add, valuenode_2};

  auto call = root_func_graph->NewCNode(call_inputs);
  auto call_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  call->set_abstract(call_abs);
  // shape
  std::vector<AnfNodePtr> shape_inputs{NewValueNode(prim::kPrimShape), call};
  auto getshape = root_func_graph->NewCNode(shape_inputs);
  auto shape_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  getshape->set_abstract(shape_abs);
  // reshape
  std::vector<AnfNodePtr> reshape_inputs{NewValueNode(prim::kPrimReshape), parameters[1], getshape};
  auto reshape = root_func_graph->NewCNode(reshape_inputs);
  auto reshape_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  reshape->set_abstract(reshape_abs);
  // return
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), reshape};
  auto root_return = root_func_graph->NewCNode(return_inputs);
  auto return_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  root_func_graph->set_return(root_return);

  // mul Graph
  // mul
  auto mulfunc_parameters = mul_func_graph->parameters();
  std::vector<AnfNodePtr> mul_inputs{NewValueNode(prim::kPrimMul), mulfunc_parameters[0], mulfunc_parameters[1]};
  auto mulfunc_node = root_func_graph->NewCNode(mul_inputs);
  auto mul_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  mulfunc_node->set_abstract(mul_abs);
  // return
  std::vector<AnfNodePtr> mulfunc_return_inputs{NewValueNode(prim::kPrimReturn), mulfunc_node};
  auto mulfunc_return = mul_func_graph->NewCNode(mulfunc_return_inputs);
  auto mulfunc_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp_4);
  mulfunc_return->set_abstract(mulfunc_return_abs);
  mul_func_graph->set_return(mulfunc_return);
  return root_func_graph;
}

FuncGraphPtr BuildGraphs() {
  auto root_func_graph = BuildFuncGraph();
  auto true_func_graph = BuildFuncGraph();
  auto false_func_graph = BuildFuncGraph();
  std::vector<int64_t> shp{2, 2};

  // root graph.
  auto parameters = root_func_graph->parameters();
  // Less.
  std::vector<AnfNodePtr> less_inputs{NewValueNode(prim::kPrimLess), parameters[0], parameters[1]};
  auto less = root_func_graph->NewCNode(less_inputs);
  AbstractTensorPtr root_less_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  less->set_abstract(root_less_abs);
  // True partial.
  std::vector<AnfNodePtr> true_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(true_func_graph),
                                              parameters[0], parameters[1]};
  auto true_partial = root_func_graph->NewCNode(true_partial_inputs);
  auto true_partial_abs = std::make_shared<FuncGraphAbstractClosure>(true_func_graph, AnalysisContext::DummyContext());
  true_partial->set_abstract(true_partial_abs);

  // False partial.
  std::vector<AnfNodePtr> false_partial_inputs{NewValueNode(prim::kPrimPartial), NewValueNode(false_func_graph),
                                               parameters[0], parameters[1]};
  auto false_partial = root_func_graph->NewCNode(false_partial_inputs);
  auto false_partial_abs =
    std::make_shared<FuncGraphAbstractClosure>(false_func_graph, AnalysisContext::DummyContext());
  false_partial->set_abstract(false_partial_abs);

  // Switch.
  std::vector<AnfNodePtr> switch_inputs{NewValueNode(prim::kPrimSwitch), less, true_partial, false_partial};
  auto switch_node = root_func_graph->NewCNode(switch_inputs);
  auto switch_abs = std::make_shared<AbstractFuncUnion>(true_partial_abs, false_partial_abs);
  switch_node->set_abstract(switch_abs);

  // Call.
  std::vector<AnfNodePtr> call_inputs{switch_node};
  auto root_call_node = root_func_graph->NewCNode(call_inputs);
  auto call_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  root_call_node->set_abstract(call_abs);
  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), root_call_node};
  auto root_return_node = root_func_graph->NewCNode(return_inputs);
  auto root_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  root_return_node->set_abstract(call_abs);
  root_func_graph->set_return(root_return_node);

  // true graph.
  auto true_parameters = true_func_graph->parameters();
  // Call.
  std::vector<AnfNodePtr> true_call_inputs{NewValueNode(root_func_graph), true_parameters[0], true_parameters[1]};
  auto true_call_node = true_func_graph->NewCNode(true_call_inputs);
  auto true_call_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_call_node->set_abstract(true_call_abs);
  // Add.
  std::vector<AnfNodePtr> true_add_inputs{NewValueNode(prim::kPrimAdd), true_parameters[0], true_call_node};
  auto true_add = true_func_graph->NewCNode(true_add_inputs);
  AbstractTensorPtr true_add_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_add->set_abstract(true_add_abs);
  // Return.
  std::vector<AnfNodePtr> true_return_inputs{NewValueNode(prim::kPrimReturn), true_add};
  auto true_return_node = true_func_graph->NewCNode(true_return_inputs);
  AbstractTensorPtr true_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  true_return_node->set_abstract(true_return_abs);
  true_func_graph->set_return(true_return_node);

  // false graph.
  // Add.
  auto false_parameters = false_func_graph->parameters();
  std::vector<AnfNodePtr> false_add_inputs{NewValueNode(prim::kPrimAdd), false_parameters[0], false_parameters[1]};
  auto false_add = false_func_graph->NewCNode(false_add_inputs);
  const auto &false_add_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  false_add->set_abstract(false_add_abs);
  // Return.
  std::vector<AnfNodePtr> false_return_inputs{NewValueNode(prim::kPrimReturn), false_add};
  auto false_return_node = false_func_graph->NewCNode(false_return_inputs);
  AbstractTensorPtr false_return_abs = std::make_shared<AbstractTensor>(kFloat32, shp);
  false_return_node->set_abstract(false_return_abs);
  false_func_graph->set_return(false_return_node);
  return root_func_graph;
}
}  // namespace

void RunTestCase(const FuncGraphPtr &func_graph) {
  const char device_name[] = "CPU";
  uint32_t device_id = 0;

  auto ms_context = MsContext::GetInstance();
  int last_execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  bool last_enable_mindrt = ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  uint32_t last_device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string last_device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device_name);

  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);
  MS_REGISTER_DEVICE(device_name, TestDeviceContext);
  DeviceContextKey device_context_key{device_name, device_id};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);

  const auto backend = std::make_shared<compile::MindRTBackend>("vm", device_name, 0);
  const auto actor_info = backend->CompileGraphs(func_graph);
  ASSERT_EQ(actor_info.find("kernel_graph") != std::string::npos, true);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, last_execution_mode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, last_enable_mindrt);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, last_device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, last_device_target);
}

/// Feature: unify runtime.
/// Description: test the compile graphs.
/// Expectation: success.
TEST_F(GraphSchedulerTest, test_singlecalltransform) { RunTestCase(BuildSingleCallFuncGraph()); }

/// Feature: unify runtime.
/// Description: test the compile graphs.
/// Expectation: success.
TEST_F(GraphSchedulerTest, test_transform) { RunTestCase(BuildGraphs()); }

FuncGraphPtr BuildAnyTypeGraph() {
  auto func_graph = BuildFuncGraph();
  MS_EXCEPTION_IF_NULL(func_graph);

  // root graph.
  auto parameters = func_graph->parameters();

  // PyExecute.
  auto script_value = MakeValue("___test_script___");
  auto script_value_node = NewValueNode(script_value);
  script_value_node->set_abstract(script_value->ToAbstract());

  auto key_value = MakeValue("___test_key___");
  auto key_value_node = NewValueNode(key_value);
  key_value_node->set_abstract(key_value->ToAbstract());
  std::vector<AnfNodePtr> pyexecute_input{NewValueNode(prim::kPrimPyExecute), script_value_node, key_value_node,
                                          parameters[0]};
  auto pyexecute_node = func_graph->NewCNode(pyexecute_input);
  auto pyexecute_abs = std::make_shared<abstract::AbstractAny>();
  pyexecute_node->set_abstract(pyexecute_abs);

  // Add.
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), pyexecute_node, parameters[1]};
  auto add_node = func_graph->NewCNode(add_inputs);
  auto add_abs = std::make_shared<abstract::AbstractAny>();
  add_node->set_abstract(add_abs);

  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), add_node};
  auto return_node = func_graph->NewCNode(return_inputs);
  auto return_abs = std::make_shared<abstract::AbstractAny>();
  return_node->set_abstract(return_abs);
  func_graph->set_return(return_node);
  return func_graph;
}

/// Feature: Pyexecute any type output.
/// Description: Test the compile of any type.
/// Expectation: As expected.
TEST_F(GraphSchedulerTest, AnyTypeKernelGraphTransform) {
  const char device_name[] = "CPU";
  uint32_t device_id = 0;

  auto ms_context = MsContext::GetInstance();
  int last_execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  bool last_enable_mindrt = ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  uint32_t last_device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string last_device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device_name);

  FuncGraphPtr func_graph = BuildAnyTypeGraph();
  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);
  MS_REGISTER_DEVICE(device_name, TestDeviceContext);
  DeviceContextKey device_context_key{device_name, device_id};
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);

  const auto backend = std::make_shared<compile::MindRTBackend>("ms", device_name, 0);
  const auto actor_info = backend->CompileGraphs(func_graph);
  ASSERT_EQ(actor_info.find("kernel_graph") != std::string::npos, true);

  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, last_execution_mode);
  ms_context->set_param<bool>(MS_CTX_ENABLE_MINDRT, last_enable_mindrt);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, last_device_id);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, last_device_target);
}
}  // namespace runtime
}  // namespace mindspore
