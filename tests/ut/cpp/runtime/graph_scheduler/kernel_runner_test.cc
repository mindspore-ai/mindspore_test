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
#include "runtime/graph_scheduler/actor/kernel_runner.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
namespace mindspore {
namespace runtime {
using namespace test;
class KernelRunnerTest : public UT::Common {
 public:
  KernelRunnerTest() {}
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

KernelGraphPtr BuildKernelGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &front_node,
                                const ValueNodePtr &prim) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto front_parameter = func_graph->parameters();

  // Build kernel.
  std::vector<AnfNodePtr> inputs{prim};
  for (const auto &parameter : front_parameter) {
    inputs.emplace_back(kernel_graph->NewParameter(parameter->cast<ParameterPtr>()));
  }
  auto backend_node = kernel_graph->NewCNode(inputs);
  std::vector<int64_t> shp{2, 2};
  abstract::AbstractTensorPtr abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  backend_node->set_abstract(abs);
  // build return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), backend_node};
  auto return_node = kernel_graph->NewCNode(return_inputs);

  kernel_graph->set_return(return_node);
  kernel_graph->set_execution_order({backend_node});
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, {front_node});
  return kernel_graph;
}
}  // namespace
/// Feature: test resetstate for uce.
/// Description: Test the parse interface.
/// Expectation: As expected.
TEST_F(KernelRunnerTest, ResetState) {
  MS_REGISTER_HAL_RES_MANAGER(kCPUDevice, DeviceType::kCPU, TestResManager);
  DeviceContextKey device_context_key{"CPU", 0};
  AID aid;

  std::set<size_t> modifiable_indexes;
  auto device_context = std::make_shared<TestDeviceContext>(device_context_key);
  MS_EXCEPTION_IF_NULL(device_context);
  auto func_graph = BuildFuncGraph();
  auto parameters = func_graph->parameters();
  // Add.
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameters[0], parameters[1]};
  auto add = func_graph->NewCNode(add_inputs);
  std::vector<int64_t> shp{2, 2};
  auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  add->set_abstract(abs);
  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), add};
  auto return_node = func_graph->NewCNode(return_inputs);
  return_node->set_abstract(abs);
  func_graph->set_return(return_node);
  // kernel graph.
  auto kernel_graph = BuildKernelGraph(func_graph, add, NewValueNode(prim::kPrimAdd));
  device_context->kernel_executor_->CreateKernel(kernel_graph->execution_order());
  auto kernel = kernel_graph->execution_order()[0];
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_runner =
    std::make_shared<runtime::KernelRunner>("", kernel, device_context.get(), aid, &aid, &aid,
                                            GraphExecutionStrategy::kPipeline, modifiable_indexes, modifiable_indexes);
  MS_EXCEPTION_IF_NULL(kernel_runner);
  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, 16, Format::DEFAULT_FORMAT, TypeId::kNumberTypeFloat32, shp, "CPU", 0);
  auto hete_info = std::make_shared<HeterogeneousInfo>();
  MS_EXCEPTION_IF_NULL(hete_info);
  hete_info->host_ptr_ = &func_graph;
  kernel_tensor->set_heterogeneous_info(hete_info);
  kernel_runner->output_kernel_tensors_.emplace_back(kernel_tensor);
  kernel_runner->ResetState();
  ASSERT_EQ(hete_info->host_ptr_, nullptr);
}
}  // namespace runtime
}  // namespace mindspore
