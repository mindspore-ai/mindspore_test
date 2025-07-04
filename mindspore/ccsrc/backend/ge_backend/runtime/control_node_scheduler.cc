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

#include "backend/ge_backend/runtime/control_node_scheduler.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "backend/ge_backend/runtime/control_node_parser.h"
#include "backend/ge_backend/runtime/scheduler_helper.h"
#include "backend/ge_backend/runtime/actor/actor_dump.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_n.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
std::string GetActorName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto debug_name = node->DebugString();
  auto index = debug_name.find('{');
  if ((index != std::string::npos) && (index > 0)) {
    debug_name = debug_name.substr(0, index);
  }

  if (common::AnfAlgo::IsCallNode(node)) {
    return "Call_" + node->UniqueName() + "_" + debug_name;
  } else {
    return node->UniqueName() + "_" + debug_name;
  }
}

std::string GetStackActorNameByExitName(const std::string &exit_name) {
  size_t pos = exit_name.find(kExitActorNameSuffix);
  if (pos == std::string::npos) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid exit actor name:" << exit_name;
  }

  return exit_name.substr(0, pos) + kStackActorNameSuffix;
}

// Parameter and ref node can not copy the device tensor.
bool is_need_copy_device_tensor(const AnfNodePtr &backend_node, size_t index) {
  MS_EXCEPTION_IF_NULL(backend_node);
  // Skip the parameter and Load node.
  const auto &real_backend_node = common::AnfAlgo::VisitKernelWithReturnType(backend_node, index, false);
  if (real_backend_node.first != nullptr && (!real_backend_node.first->isa<CNode>())) {
    return false;
  }
  auto kernel_graph = AnfAlgo::FetchKernelGraph(backend_node.get());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->IsInRefOutputMap(real_backend_node)) {
    const auto &origin_node = kernel_graph->GetRefNodeRecursive(real_backend_node).first;
    MS_EXCEPTION_IF_NULL(origin_node);
    if (origin_node->isa<ValueNode>() || origin_node->isa<Parameter>()) {
      return false;
    }
  }
  return true;
}

// Check whether the exit actor corresponding to the call node to the to actor already exists control arrow.
bool IsControlArrowExistForCallNode(const AnfNodePtr &node, const AbstractActor *const to_actor,
                                    const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(parser);
  if (!common::AnfAlgo::IsCallNode(node)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
      << "#dmsg#Runtime error info:#dmsg#Invalid call node:" << node->DebugString();
  }
  int branch_id = parser->FetchBranchIDByCallNode(node);

  const auto &func_graphs = parser->FetchFuncGraphbyCallNode(node);
  if (func_graphs.empty()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
      << "#dmsg#Runtime error info:#dmsg#Failed to get funcgraph by call node:" << node->DebugString();
  }
  MS_EXCEPTION_IF_NULL(*(func_graphs.begin()));
  auto actor_name = (*(func_graphs.begin()))->ToString() + kExitActorNameSuffix;
  const auto &actor = FetchActor(actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);

  const auto &branch_arrows = exit_actor->output_branch_control_arrows();
  const auto &arrow_iter = branch_arrows.find(branch_id);
  if (arrow_iter == branch_arrows.end()) {
    return false;
  }
  const auto &arrows = arrow_iter->second;
  return std::find(arrows.begin(), arrows.end(), to_actor->GetAID()) != arrows.end();
}

bool IsNotCut(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return cnode->HasPrimalAttr(kAttrNotCut);
}
}  // namespace

ControlActorSetPtr ControlNodeScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                               const AID &memory_manager_aid) {
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  if (control_nodes.size() <= kSingleControlNode) {
    return nullptr;
  }

  memory_manager_aid_ = memory_manager_aid;
  ControlActorSetPtr control_actors = std::make_shared<ControlActorSet>();
  MS_EXCEPTION_IF_NULL(control_actors);
  control_actors->switch_actors_ = BuildSwitchActor(graph_compiler_info);
  control_actors->gather_actors_ = BuildGatherActor(graph_compiler_info);
  control_actors->entrance_actors_ = BuildEntranceActor(graph_compiler_info);
  control_actors->exit_actors_ = BuildExitActor(graph_compiler_info);
  control_actors->stack_actors_ = BuildStackActor(graph_compiler_info);
  return control_actors;
}

std::vector<SwitchActorPtr> ControlNodeScheduler::BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<SwitchActorPtr> switch_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;

  for (const auto &control_node : control_nodes) {
    // Switch node and switch layer node will be converted to switch actor.
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &switch_actor =
        std::make_shared<SwitchActor>(actor_name, memory_manager_aid_, parameters, control_node);
      (void)switch_actors.emplace_back(switch_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }
      InsertActor(switch_actor.get());
    }
  }
  return switch_actors;
}

void ControlNodeScheduler::BuildDataSourceActorForControlNode(
  const GraphCompilerInfo &graph_compiler_info, const HostTensorQueuePtr &host_queue,
  const HostQueueDSActorPtr &host_queue_ds_actor, const AID &memory_manager_aid,
  std::vector<DataSourceActorPtr> *data_source_actors) const {
  HostQueueDSActorPtr control_node_ds_actor = host_queue_ds_actor;
  const auto parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(data_source_actors);

  // Initialize the parameter in the control node, first get all the front parameters in the control node, then find
  // the corresponding backend parameter from the map, and insert it into the host data source actor.
  const auto &control_node_parameters = parser->control_node_parameters();
  for (const auto &parameter_with_index : control_node_parameters) {
    MS_EXCEPTION_IF_NULL(parameter_with_index.first);
    if (IsPersistentDeviceTensor(parameter_with_index.first)) {
      continue;
    }
    if (control_node_ds_actor == nullptr) {
      auto actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
      MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
      control_node_ds_actor = std::make_shared<HostQueueDataSourceActor>(
        actor_name, 1, memory_manager_aid, nullptr, nullptr, host_queue, graph_compiler_info.graph_phase_);
      MS_EXCEPTION_IF_NULL(control_node_ds_actor);
      InsertActor(control_node_ds_actor.get());
      (void)data_source_actors->emplace_back(control_node_ds_actor);
    }

    auto &node_map = control_node_ds_actor->data_node_position_map_;
    if (node_map.find(parameter_with_index) != node_map.end()) {
      continue;
    }
    graph_compiler_info.origin_parameters_to_backend_parameters_[parameter_with_index.first].emplace_back(
      std::make_pair(parameter_with_index, parameter_with_index));

    const auto &node_with_index = parser->FetchBackendParameterWithContextByFrontParameter(parameter_with_index);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);

    MS_EXCEPTION_IF_NULL(node_with_index.first);
    MS_LOG(DEBUG) << "Control node parameter front node:" << parameter_with_index.first->DebugString()
                  << " index:" << parameter_with_index.second
                  << " backend node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    auto iter = find(control_node_ds_actor->data_node_with_indexs_.begin(),
                     control_node_ds_actor->data_node_with_indexs_.end(), node_with_index);
    if (iter != control_node_ds_actor->data_node_with_indexs_.end()) {
      (void)node_map.emplace(parameter_with_index, iter - control_node_ds_actor->data_node_with_indexs_.begin());
      MS_LOG(DEBUG) << "Insert front node:" << parameter_with_index.first->DebugString()
                    << " index:" << parameter_with_index.second << " to host queue data source actor.";
    } else {
      CreateBuildInfoForFrontNode(parameter_with_index, node_with_index.first);
      // Create device tensor.
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(node_with_index.first, node_with_index.second, false);
      MS_EXCEPTION_IF_NULL(device_address);
      const auto &sub_abstract =
        common::AnfAlgo::FetchAbstractByIndex(parameter_with_index.first->abstract(), parameter_with_index.second);
      MS_EXCEPTION_IF_NULL(sub_abstract);
      const auto &kernel_tensor = AnfAlgo::CreateKernelTensor(
        sub_abstract->BuildShape(), sub_abstract->BuildType(), nullptr, nullptr, device_address->GetSize(),
        device_address->format(), device_address->type_id(), device_address->host_shape(), device_name, device_id);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(parameter_with_index.first));

      auto new_address = kernel_tensor->device_address().get();
      MS_EXCEPTION_IF_NULL(new_address);
      MS_LOG(DEBUG) << "Create new address for node that has no corresponding backend node:"
                    << parameter_with_index.first->DebugString() << " index:" << parameter_with_index.second
                    << " addr:" << new_address << " size:" << device_address->GetSize()
                    << ", type id:" << device_address->type_id()
                    << " type:" << (kernel_tensor->GetType() == nullptr ? "null" : kernel_tensor->GetType()->ToString())
                    << " shape:"
                    << (kernel_tensor->GetShape() == nullptr ? "null" : kernel_tensor->GetShape()->ToString());
      AnfAlgo::SetOutputKernelTensor(kernel_tensor, parameter_with_index.second, parameter_with_index.first.get());

      (void)node_map.emplace(parameter_with_index, control_node_ds_actor->data_node_with_indexs_.size());
      (void)control_node_ds_actor->data_node_with_indexs_.emplace_back(parameter_with_index);
    }
  }
}

std::vector<GatherActorPtr> ControlNodeScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<GatherActorPtr> gather_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    // Partial node and call node will be converted to gather actor.
    if ((common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial) && (!IsInvalidPartial(control_node))) ||
        common::AnfAlgo::IsCallNode(control_node)) {
      const auto &actor_name = GetActorName(control_node);
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &gather_actor =
        std::make_shared<GatherActor>(actor_name, memory_manager_aid_, parameters, control_node);
      MS_EXCEPTION_IF_NULL(gather_actor);
      (void)gather_actors.emplace_back(gather_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }
      InsertActor(gather_actor.get());

      // The gather actor corresponding to a call node needs to set the branch id.
      if (common::AnfAlgo::IsCallNode(control_node)) {
        gather_actor->output_branch_id_ = parser->FetchBranchIDByCallNode(control_node);
      }
    }
  }
  return gather_actors;
}

std::vector<EntranceActorPtr> ControlNodeScheduler::BuildEntranceActor(
  const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &call_node_to_func_graphs = parser->call_node_to_func_graphs_;
  std::unordered_map<FuncGraphPtr, std::set<KernelWithIndex>> func_graph_to_call_nodes;
  for (const auto &call_node_to_func_graph : call_node_to_func_graphs) {
    const auto &node = call_node_to_func_graph.first;
    for (const auto &func_graph : call_node_to_func_graph.second) {
      (void)func_graph_to_call_nodes[func_graph].emplace(node, 0);
    }
  }

  std::vector<EntranceActorPtr> entrance_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
      std::vector<KernelWithIndex> formal_parameters;

      // The entrance actor has two parts of node members :
      // 1. The formal parameters of the subgraph are used to connect the actor's output arrows.
      for (const auto &parameter : func_graph->parameters()) {
        MS_EXCEPTION_IF_NULL(parameter);
        const auto &abstract = parameter->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
        for (size_t i = 0; i < output_num; ++i) {
          (void)formal_parameters.emplace_back(parameter, i);
        }
      }

      // 2. The caller of the subgraph, namely call nodes, is used to connect the input arrows.
      std::set<KernelWithIndex> call_nodes;
      const auto &iter = func_graph_to_call_nodes.find(func_graph);
      if (iter != func_graph_to_call_nodes.end()) {
        call_nodes = iter->second;
      }
      for (const auto &formal_parameter : formal_parameters) {
        MS_EXCEPTION_IF_NULL(formal_parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << formal_parameter.first->DebugString() << " index:" << formal_parameter.second;
      }
      const auto &entrance_actor =
        std::make_shared<EntranceActor>(actor_name, memory_manager_aid_, formal_parameters, call_nodes, control_node);
      MS_EXCEPTION_IF_NULL(entrance_actor);
      (void)entrance_actors.emplace_back(entrance_actor);
      InsertActor(entrance_actor.get());
    }
  }

  return entrance_actors;
}

std::vector<ExitActorPtr> ControlNodeScheduler::BuildExitActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<ExitActorPtr> exit_actors;
  const auto &control_nodes = graph_compiler_info.control_nodes_;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // The exit actor is used in 2 places:
  // 1.funcgraph output, that is the output of the return node.
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      const auto &parameters = FetchInputNodeByCNode(control_node);
      const auto &exit_actor = std::make_shared<ExitActor>(actor_name, memory_manager_aid_, parameters, control_node);
      MS_EXCEPTION_IF_NULL(exit_actor);
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter.first);
        MS_LOG(DEBUG) << "Print formal parameter for actor:" << actor_name
                      << " parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
      }

      (void)exit_actors.emplace_back(exit_actor);
      InsertActor(exit_actor.get());
    }
  }

  // 2. Replace the device address in the kernel actor when calling funcgraph, that is to say in the data exchange
  // between kernel graph and the control node, in fact, it is the output of the kernel graph.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    if (kernel_graph_group_info->graphs_.empty()) {
      continue;
    }

    std::vector<bool> is_need_copy_device_tensors;
    std::vector<bool> is_need_dynamic_checks;
    std::vector<bool> is_dynamic_shapes;
    std::vector<KernelWithIndex> formal_parameters;

    for (const auto &node_with_context : kernel_graph_group_info->front_output_nodes_) {
      if (HasAbstractMonad(node_with_context.first.first) || IsCsrNode(node_with_context.first.first)) {
        continue;
      }
      // Collect inputs of exit actor.
      (void)formal_parameters.emplace_back(node_with_context.first);
      // Get the device contexts of the exit actor's cnode inputs.
      const AnfNodePtr &backend_node = node_with_context.second.first;
      MS_EXCEPTION_IF_NULL(backend_node);
      (void)is_need_copy_device_tensors.emplace_back(
        is_need_copy_device_tensor(backend_node, node_with_context.second.second));
      (void)is_need_dynamic_checks.emplace_back(
        common::AnfAlgo::CheckPrimitiveType(backend_node, prim::kPrimConditionGather));
      auto is_dynamic_shape =
        common::AnfAlgo::IsDynamicShape(backend_node) || common::AnfAlgo::IsDynamicSequence(backend_node);
      (void)is_dynamic_shapes.emplace_back(is_dynamic_shape);
    }
    const auto &actor_name = kernel_graph_group_info->group_name_ + kExitActorNameSuffix;
    const auto &exit_actor = std::make_shared<ExitActor>(actor_name, memory_manager_aid_, formal_parameters, nullptr);
    MS_EXCEPTION_IF_NULL(exit_actor);
    exit_actor->is_need_copy_device_tensors_.swap(is_need_copy_device_tensors);
    exit_actor->is_need_dynamic_checks_.swap(is_need_dynamic_checks);
    exit_actor->is_dynamic_shapes_.swap(is_dynamic_shapes);
    for (const auto &graph : kernel_graph_group_info->graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
      std::for_each(graph->GetRefMap().begin(), graph->GetRefMap().end(),
                    [&exit_actor, &graph](const std::pair<KernelWithIndex, KernelWithIndex> &pair) {
                      exit_actor->ref_out_in_map_[pair.first] = graph->GetRefNodeRecursive(pair.first);
                    });
    }
    (void)exit_actors.emplace_back(exit_actor);
    InsertActor(exit_actor.get());
  }

  return exit_actors;
}

std::vector<StackActorPtr> ControlNodeScheduler::BuildStackActor(const GraphCompilerInfo &graph_compiler_info) const {
  std::vector<StackActorPtr> stack_actors;
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Create a corresponding stack actor for each kernel graph that has a call node as input.
  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    if (!kernel_graph_group_info->need_stack_) {
      continue;
    }
    const auto &actor_name = kernel_graph_group_info->group_name_ + kStackActorNameSuffix;
    size_t input_parameter_data_num = 0;
    std::vector<KernelWithIndex> formal_parameters;
    // Collect inputs of stack actor.
    for (const auto &node_with_context : kernel_graph_group_info->front_input_nodes_) {
      // If the input comes from inside funcgraph, put it at the front of the vector, otherwise put it at the end.
      const auto &from_node = node_with_context.first;
      MS_EXCEPTION_IF_NULL(from_node);
      auto iter = parser->node_to_level_.find(from_node);
      if (iter == parser->node_to_level_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, from_node)
          << "#dmsg#Runtime error info:#dmsg#Failed to get level by from node:" << from_node->DebugString()
          << " in graph:" << kernel_graph_group_info->group_name_;
      }
      if (iter->second == kernel_graph_group_info->level_ && (!parser->IsRootGraphPersistentDeviceTensor(from_node))) {
        (void)formal_parameters.emplace_back(node_with_context);
        MS_LOG(DEBUG) << "Add normal parameter for actor:" << actor_name << " node:" << from_node->DebugString()
                      << " index:" << node_with_context.second;
      } else {
        (void)formal_parameters.insert(formal_parameters.begin(), node_with_context);
        MS_LOG(DEBUG) << "Add stack parameter for actor:" << actor_name << " node:" << from_node->DebugString()
                      << " index:" << node_with_context.second;
        input_parameter_data_num++;
      }
    }
    const auto &stack_actor = std::make_shared<StackActor>(actor_name, memory_manager_aid_, formal_parameters);
    MS_EXCEPTION_IF_NULL(stack_actor);
    (void)stack_actors.emplace_back(stack_actor);
    stack_actor->input_stack_data_num_ = input_parameter_data_num;
    InsertActor(stack_actor.get());
  }
  // Create stack actors for control nodes.
  BuildStackActorForControlNode(graph_compiler_info, &stack_actors);

  return stack_actors;
}

void ControlNodeScheduler::BuildStackActorForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                                         std::vector<StackActorPtr> *const stack_actors) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &need_stack_control_node : parser->need_stack_control_nodes_) {
    MS_EXCEPTION_IF_NULL(need_stack_control_node);
    MS_LOG(DEBUG) << "Build stack actor for control node:" << need_stack_control_node->DebugString();
    const auto &stack_actor_name = GetActorName(need_stack_control_node) + kStackActorNameSuffix;
    std::vector<KernelWithIndex> formal_parameters;
    size_t input_parameter_data_num = 0;
    size_t input_parameter_partials_num = 0;

    // Fetch the control actor of control node.
    std::string control_actor_name = "";
    if (common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimReturn)) {
      const auto &func_graph = need_stack_control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      control_actor_name = func_graph->ToString() + kExitActorNameSuffix;
    } else if (common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimPartial) ||
               common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitch) ||
               common::AnfAlgo::CheckPrimitiveType(need_stack_control_node, prim::kPrimSwitchLayer) ||
               common::AnfAlgo::IsCallNode(need_stack_control_node)) {
      control_actor_name = GetActorName(need_stack_control_node);
    } else {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, need_stack_control_node)
        << "#dmsg#Runtime error info:#dmsg#Invalid control node:" << need_stack_control_node->DebugString();
    }

    auto iter = parser->node_to_level_.find(need_stack_control_node);
    if (iter == parser->node_to_level_.end()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, need_stack_control_node)
        << "#dmsg#Runtime error info:#dmsg#Failed to get level for need stack control node:"
        << need_stack_control_node->DebugString();
    }
    size_t control_node_level = iter->second;

    auto actor = FetchActor(control_actor_name);
    if (actor == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid actor name:" << control_actor_name;
    }
    auto control_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(control_actor);

    // Collect formal parameters and device contexts, skip the value nodes.
    for (size_t i = 0; i < control_actor->formal_parameters_.size(); ++i) {
      const auto &parameter = control_actor->formal_parameters_[i];
      MS_EXCEPTION_IF_NULL(parameter.first);
      if (parameter.first->isa<ValueNode>()) {
        continue;
      }

      iter = parser->node_to_level_.find(parameter.first);
      if (iter == parser->node_to_level_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, parameter.first)
          << "#dmsg#Runtime error info:#dmsg#Failed to get level for formal parameter:"
          << parameter.first->DebugString()
          << " for need stack control node:" << need_stack_control_node->DebugString();
      }

      if (control_node_level == iter->second && (!parser->IsRootGraphPersistentDeviceTensor(parameter.first))) {
        MS_LOG(DEBUG) << "Add normal parameter:" << parameter.first->DebugString()
                      << " for stack actor:" << stack_actor_name;
        (void)formal_parameters.emplace_back(parameter);
      } else {
        MS_LOG(DEBUG) << "Add stack parameter:" << parameter.first->DebugString()
                      << " for stack actor:" << stack_actor_name;
        (void)formal_parameters.insert(formal_parameters.begin(), parameter);

        const auto &abstract = parameter.first->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, parameter.second);
        MS_EXCEPTION_IF_NULL(real_abstract);
        if (real_abstract->isa<abstract::AbstractFunction>()) {
          input_parameter_partials_num++;
        } else {
          input_parameter_data_num++;
        }
      }
    }
    // Create stack actor.
    const auto &stack_actor = std::make_shared<StackActor>(stack_actor_name, memory_manager_aid_, formal_parameters);
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->input_stack_data_num_ = input_parameter_data_num;
    stack_actor->input_stack_partials_num_ = input_parameter_partials_num;
    stack_actor->node_ = need_stack_control_node;
    InsertActor(stack_actor.get());
    (void)stack_actors->emplace_back(stack_actor);
  }
}

namespace {
void ParseRealIndex(const mindspore::HashMap<size_t, size_t> &dynamic_len_index, size_t formal_input_num,
                    std::vector<std::pair<std::vector<size_t>, bool>> *real_indexes, AbstractActor *actor) {
  MS_EXCEPTION_IF_NULL(real_indexes);
  MS_EXCEPTION_IF_NULL(actor);
  auto tmp_dynamic_len_index = dynamic_len_index;
  size_t real_output_num = formal_input_num + tmp_dynamic_len_index.size();
  for (const auto &index_pair : tmp_dynamic_len_index) {
    if (real_output_num < index_pair.second) {
      MS_LOG(EXCEPTION) << "Invalid dynamic len:" << std::to_string(index_pair.second)
                        << " start index:" << std::to_string(index_pair.first)
                        << " real input num:" << std::to_string(real_output_num)
                        << " for actor:" << actor->GetAID().Name();
    }
    real_output_num -= index_pair.second;
  }
  MS_LOG(DEBUG) << "for actor:" << actor->GetAID() << " real output num:" << real_output_num;
  size_t start_index = 0;
  for (const auto &pair : tmp_dynamic_len_index) {
    MS_LOG(DEBUG) << "start index:" << pair.first << " len:" << pair.second;
  }
  for (size_t i = 0; i < real_output_num; ++i) {
    MS_LOG(DEBUG) << "for actor:" << actor->GetAID() << " real output index:" << i;
    if (tmp_dynamic_len_index.find(start_index) != tmp_dynamic_len_index.end()) {
      std::vector<size_t> indexes;
      for (size_t j = 0; j < tmp_dynamic_len_index[start_index]; ++j) {
        indexes.emplace_back(j + start_index);
      }
      real_indexes->emplace_back(indexes, true);
      size_t old_index = start_index;
      start_index += tmp_dynamic_len_index[start_index];
      tmp_dynamic_len_index.erase(old_index);
    } else {
      std::vector<size_t> indexes{start_index};
      real_indexes->emplace_back(indexes, false);
      start_index++;
    }
  }
  for (size_t i = 0; i < real_indexes->size(); ++i) {
    std::string index_str = "index_";
    for (const auto &index : (*real_indexes)[i].first) {
      index_str = index_str + std::to_string(index) + "_";
    }
    MS_LOG(DEBUG) << "for actor:" << actor->GetAID() << " real input " << i << " " << index_str;
  }
  if (real_indexes->size() != real_output_num) {
    MS_LOG(EXCEPTION) << "Invalid real index size:" << std::to_string(real_indexes->size())
                      << " start need:" << std::to_string(real_output_num) << " for actor:" << actor->GetAID().Name();
  }
}
}  // namespace

void ControlNodeScheduler::CollectDynamicLenIndexForArgment(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  for (const auto &node_to_func_with_index : parser->control_node_to_funcgraph_with_dynamic_sequence_index_) {
    const auto &node = node_to_func_with_index.first;
    MS_EXCEPTION_IF_NULL(node);
    const auto &actor_name = GetActorName(node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    for (const auto &func_with_index : node_to_func_with_index.second) {
      const auto &func_graph = func_with_index.first;
      MS_EXCEPTION_IF_NULL(func_graph);
      auto dynamic_len_index = func_with_index.second;
      std::vector<std::pair<std::vector<size_t>, bool>> real_indexes;
      ParseRealIndex(dynamic_len_index, gather_actor->formal_parameters_.size(), &real_indexes, gather_actor);
      MS_LOG(INFO) << "Add dynamic len index for funcgraph:" << func_graph->ToString()
                   << " actor:" << gather_actor->GetAID()
                   << " formal parameter num:" << gather_actor->formal_parameters_.size();
      gather_actor->dynamic_len_index_[func_graph] = real_indexes;
    }
  }

  for (const auto &node_to_call_with_index : parser->return_to_call_with_dynamic_sequence_index_) {
    const auto &node = node_to_call_with_index.first;
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());
    MS_LOG(DEBUG) << "for node:" << node->DebugString();
    const auto &actor_name = node->func_graph()->ToString() + kExitActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
    MS_EXCEPTION_IF_NULL(exit_actor);
    for (const auto &call_with_index : node_to_call_with_index.second) {
      const auto &call = call_with_index.first;
      MS_EXCEPTION_IF_NULL(call);
      int branch_id = parser->FetchBranchIDByCallNode(call);
      std::vector<std::pair<std::vector<size_t>, bool>> real_indexes;
      ParseRealIndex(call_with_index.second, exit_actor->formal_parameters_.size(), &real_indexes, exit_actor);
      exit_actor->output_branch_dynamic_len_index_[branch_id] = real_indexes;
    }
  }
}

void ControlNodeScheduler::Link(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);
  MS_LOG(DEBUG) << "Control node scheduler link start.";
  // Link data arrows and partial arrows between control actors.
  LinkArrowForControlActor(actor_set->control_actors_.get(), graph_compiler_info);

  // Link arrows from host data source actor or data prepare actor to entrance actor of root graph.
  LinkArrowForRootGraphEntranceActor(actor_set, graph_compiler_info);

  // Link output data arrows from control actors to output actor.
  LinkDataArrowForOutputActor(actor_set, graph_compiler_info);

  // Link data arrows from entrance actors to kernel actors.
  LinkDataArrowForKernelActor(graph_compiler_info);

  // Link branch id arrows between control actors.
  LinkBranchIDArrowForControlActor(actor_set->control_actors_.get());

  // Link all control arrows between control actors.
  LinkControlArrowForControlActor(actor_set, graph_compiler_info);

  // Link control arrows for no input and no output kernel actor.
  LinkControlArrowForKernelActor(actor_set, graph_compiler_info);

  LinkControlArrowForLoopCountActor(actor_set, graph_compiler_info);

  SetTimeSummaryForControlActor(graph_compiler_info);

  CollectDynamicLenIndexForArgment(graph_compiler_info);
  MS_LOG(DEBUG) << "Control node scheduler link end.";
}

void ControlNodeScheduler::ClearActorData(const ControlActorSet *control_actor_set) const {
  if (control_actor_set == nullptr) {
    return;
  }

  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    switch_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    gather_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
    gather_actor->created_kernel_tensors_.clear();
    gather_actor->created_new_graphs_.clear();
    gather_actor->created_new_nodes_.clear();
  }

  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    entrance_actor->created_kernel_tensors_.clear();
    entrance_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->created_kernel_tensors_.clear();
    stack_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    exit_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
    exit_actor->last_step_created_kernel_tensors_.swap(exit_actor->created_kernel_tensors_);
    exit_actor->created_new_graphs_.clear();
    exit_actor->created_new_nodes_.clear();
  }
}

namespace {
FuncGraphPtr GetLazyInlineFuncGraph(const StackActorPtr &stack_actor) {
  MS_EXCEPTION_IF_NULL(stack_actor);
  for (const auto &input_pair : stack_actor->input_data_arrow_aids()) {
    if (input_pair.second == nullptr) {
      continue;
    }
    const auto &data_arrow = input_pair.second;
    if (IntToSize(data_arrow->to_input_index_) < stack_actor->input_stack_data_num()) {
      continue;
    }
    std::string actor_name = input_pair.first.Name();
    if (actor_name.empty()) {
      continue;
    }
    const auto &actor = FetchActor(actor_name);
    if (actor == nullptr) {
      MS_LOG(WARNING) << "Failed to get actor by aid:" << actor_name << " for stack actor:" << stack_actor->GetAID();
      continue;
    }
    if (actor->type() != KernelTransformType::kExitActor) {
      continue;
    }
    const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
    MS_EXCEPTION_IF_NULL(exit_actor);
    if (exit_actor->node() == nullptr) {
      continue;
    }
    return exit_actor->node()->func_graph();
  }
  return nullptr;
}
}  // namespace

void ControlNodeScheduler::OptimizeBranchIdArrow(const ActorSetPtr &actor_set,
                                                 const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;

  for (const auto &entrance_actor : actor_set->control_actors_->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    std::stable_partition(entrance_actor->output_data_arrows_.begin(), entrance_actor->output_data_arrows_.end(),
                          [](const DataArrowPtr &arrow) {
                            MS_EXCEPTION_IF_NULL(arrow);
                            const auto &actor = FetchActor(arrow->to_op_id_.Name());
                            return actor != nullptr && (actor->type() == KernelTransformType::kSuperKernelActor);
                          });
  }

  for (const auto &stack_actor : actor_set->control_actors_->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    if (stack_actor->formal_parameters_.size() !=
        stack_actor->input_data_arrow_aids_.size() + stack_actor->device_tensor_store_keys_.size()) {
      continue;
    }
    if (stack_actor->input_stack_data_num_ == 0 ||
        stack_actor->input_stack_data_num_ >= stack_actor->device_tensor_store_keys_.size() +
                                                stack_actor->local_kernel_tensors_.size() +
                                                stack_actor->input_data_arrow_aids_.size()) {
      continue;
    }
    const auto &func_graph = GetLazyInlineFuncGraph(stack_actor);
    if (func_graph == nullptr) {
      continue;
    }
    if (!func_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      continue;
    }
    const auto &iter = parser->func_graph_to_call_nodes_.find(func_graph);
    if (iter != parser->func_graph_to_call_nodes_.end() && (!iter->second.empty())) {
      continue;
    }
    for (const auto &input_aid : stack_actor->input_branch_id_arrow_aids_) {
      const auto &actor = FetchActor(input_aid.Name());
      if (actor == nullptr || actor->type() != KernelTransformType::kEntranceActor) {
        MS_LOG(WARNING) << "Invalid input branch id aid:" << input_aid;
        continue;
      }
      const auto &entrance_actor = dynamic_cast<EntranceActor *>(actor);
      MS_EXCEPTION_IF_NULL(entrance_actor);
      const auto &branch_id_iter = std::find(entrance_actor->output_branch_id_arrows_.begin(),
                                             entrance_actor->output_branch_id_arrows_.end(), stack_actor->GetAID());
      if (branch_id_iter != entrance_actor->output_branch_id_arrows_.end()) {
        stack_actor->is_branch_id_enable_ = false;
        entrance_actor->output_branch_id_arrows_.erase(branch_id_iter);
        MS_LOG(DEBUG) << "Disable branch id from entrance actor:" << entrance_actor->GetAID()
                      << " to stack actor:" << stack_actor->GetAID()
                      << " for cell reuse funcgraph:" << func_graph->ToString();
        break;
      }
    }
  }
}

void ControlNodeScheduler::CollectIgnoreIndexForEntranceActor(std::set<int> *ignore_index,
                                                              const EntranceActorPtr &entrance_actor) const {
  MS_EXCEPTION_IF_NULL(ignore_index);
  MS_EXCEPTION_IF_NULL(entrance_actor);
  mindspore::HashMap<int, bool> from_index_to_ignore;
  for (size_t i = 0; i < entrance_actor->output_data_arrows().size(); ++i) {
    const auto &data_arrow = entrance_actor->output_data_arrows()[i];
    int from_index = data_arrow->from_output_index_;
    if (from_index_to_ignore.find(from_index) == from_index_to_ignore.end()) {
      from_index_to_ignore[from_index] = entrance_actor->output_need_disable_dynamic_ref_counts_[i];
    } else {
      from_index_to_ignore[from_index] =
        (from_index_to_ignore[from_index] && entrance_actor->output_need_disable_dynamic_ref_counts_[i]);
    }
    MS_LOG(DEBUG) << "from index:" << from_index << " bool:" << from_index_to_ignore[from_index];
  }
  for (const auto &pair : from_index_to_ignore) {
    MS_LOG(DEBUG) << "Actor:" << entrance_actor->GetAID() << " from index:" << pair.first
                  << " is ignore:" << pair.second;
    if ((!pair.second) || pair.first < 0 || (IntToSize(pair.first) >= entrance_actor->formal_parameters_.size())) {
      continue;
    }
    ignore_index->emplace(pair.first);
    MS_LOG(INFO) << "Add ignore index:" << pair.first
                 << " by paraemter for entrance actor:" << entrance_actor->GetAID();
  }
}

bool ControlNodeScheduler::CheckIsValidArgIndex(size_t index, const EntranceActorPtr &entrance_actor,
                                                const ControlActor *gather_actor, const FuncGraphPtr &func_graph,
                                                const CNodePtr &partial_cnode, size_t *to_index) const {
  MS_EXCEPTION_IF_NULL(entrance_actor);
  MS_EXCEPTION_IF_NULL(gather_actor);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(partial_cnode);
  MS_EXCEPTION_IF_NULL(to_index);
  if (index >= entrance_actor->formal_parameters_.size() ||
      entrance_actor->formal_parameters_[index].first == nullptr) {
    MS_LOG(WARNING) << "Invalid ignore index:" << index
                    << " total parameter num:" << entrance_actor->formal_parameters_.size()
                    << " for actor:" << entrance_actor->GetAID();
    return false;
  }
  if (entrance_actor->formal_parameters_[index].first->abstract() == nullptr ||
      entrance_actor->formal_parameters_[index].first->abstract()->isa<abstract::AbstractSequence>()) {
    MS_LOG(INFO) << "Invalid abstract for parameter:" << entrance_actor->formal_parameters_[index].first->DebugString()
                 << " in actor:" << entrance_actor->GetAID();
    return false;
  }
  auto iter = std::find(func_graph->parameters().begin(), func_graph->parameters().end(),
                        entrance_actor->formal_parameters_[index].first);
  if (iter == func_graph->parameters().end()) {
    MS_LOG(WARNING) << "Failed to get index for parameter:"
                    << entrance_actor->formal_parameters_[index].first->DebugString()
                    << " in actor:" << entrance_actor->GetAID();
    return false;
  }
  size_t relative_index = LongToSize(iter - func_graph->parameters().begin());
  if (relative_index >= partial_cnode->size() - kPartialInputStartPos) {
    MS_LOG(WARNING) << "Relative index:" << relative_index
                    << " is exceed the input size of partial:" << partial_cnode->DebugString();
    return false;
  }
  const auto &arg_node = partial_cnode->input(relative_index + kPartialInputStartPos);
  if (arg_node == nullptr) {
    MS_LOG(WARNING) << "Invalid arg node of index:" << relative_index
                    << " in partial node:" << partial_cnode->DebugString();
    return false;
  }
  if (arg_node->abstract() == nullptr || arg_node->abstract()->isa<abstract::AbstractSequence>()) {
    MS_LOG(INFO) << "Invalid abstract for arg node:" << arg_node->DebugString()
                 << " in partial node:" << partial_cnode->DebugString();
    return false;
  }
  const auto &arg_iter = std::find_if(gather_actor->formal_parameters_.begin(), gather_actor->formal_parameters_.end(),
                                      [arg_node](const auto &pair) { return pair.first == arg_node; });
  if (arg_iter == gather_actor->formal_parameters_.end()) {
    MS_LOG(INFO) << "Failed to get arg node:" << arg_node->DebugString()
                 << " in gather actor:" << gather_actor->GetAID();
    return false;
  }
  *to_index = arg_iter - gather_actor->formal_parameters_.begin();
  return true;
}

void ControlNodeScheduler::Optimize(const ActorSetPtr &actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  return;
}

void ControlNodeScheduler::LinkArrowForControlActor(ControlActorSet *const control_actor_set,
                                                    const GraphCompilerInfo &graph_compiler_info) const {
  if (control_actor_set == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (!parser->IsNeedStackControlNode(switch_actor->node_)) {
      for (size_t i = 0; i < switch_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(switch_actor.get(), switch_actor->formal_parameters_[i], {switch_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(switch_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, switch_actor.get(), graph_compiler_info);
    }
  }

  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    MS_EXCEPTION_IF_NULL(gather_actor->node_);
    if (!parser->IsNeedStackControlNode(gather_actor->node_)) {
      for (size_t i = 0; i < gather_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(gather_actor.get(), gather_actor->formal_parameters_[i], {gather_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor_name = GetActorName(gather_actor->node_) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, gather_actor.get(), graph_compiler_info);
    }
  }

  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    for (const auto &call_node : entrance_actor->call_nodes_) {
      LinkArrowbyFormalParameter(entrance_actor.get(), call_node, {entrance_actor->node_, 0}, graph_compiler_info);
    }
  }

  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);

    auto stack_actor_name = (exit_actor->node_ == nullptr ? GetStackActorNameByExitName(exit_actor->GetAID().Name())
                                                          : GetActorName(exit_actor->node_) + kStackActorNameSuffix);
    auto actor = FetchActor(stack_actor_name);
    if (actor == nullptr) {
      for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
        LinkArrowbyFormalParameter(exit_actor.get(), exit_actor->formal_parameters_[i], {exit_actor->node_, i},
                                   graph_compiler_info);
      }
    } else {
      // If the control actor has a corresponding stack actor, the input should be linked to the stack actor.
      auto stack_actor = dynamic_cast<StackActor *>(actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      LinkArrowFromStackActor(stack_actor, exit_actor.get(), graph_compiler_info);
    }
  }

  for (auto &stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    for (size_t i = 0; i < stack_actor->formal_parameters_.size(); ++i) {
      LinkArrowbyFormalParameter(stack_actor.get(), stack_actor->formal_parameters_[i], {stack_actor->node_, i},
                                 graph_compiler_info);
    }
  }
}

void ControlNodeScheduler::LinkArrowFromStackActor(StackActor *const stack_actor, ControlActor *const to_actor,
                                                   const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(stack_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (size_t to_index = 0; to_index < to_actor->formal_parameters_.size(); ++to_index) {
    const auto &formal_parameter =
      common::AnfAlgo::FetchRealNodeSkipMonadControl(to_actor->formal_parameters_[to_index]);
    const auto &from_node = formal_parameter.first;
    MS_EXCEPTION_IF_NULL(from_node);
    if (from_node->isa<ValueNode>()) {
      LinkArrowByValueNode(from_node, to_actor, formal_parameter.second, to_index);
      continue;
    }

    // Fetch the arrow type of input.
    if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr && from_node->isa<CNode>() &&
        (!common::AnfAlgo::IsCallNode(from_node) || IsNotCut(from_node)) &&
        (!common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) &&
        to_actor->GetAID().Name().find(
          parser->FetchGroupNameByKernelGraph(parser->FetchKernelGraphByFrontNode(from_node))) != std::string::npos) {
      LinkArrowByKernel(from_node, to_actor, formal_parameter, {to_actor->node_, to_index}, graph_compiler_info);
      continue;
    }

    size_t from_index = stack_actor->FetchNodePosition(formal_parameter);
    const auto &abstract = from_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, formal_parameter.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    // Link arrow according to abstract.
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      SchedulerHelper::AddPartialArrow(stack_actor, to_actor, from_index, to_index);
    } else {
      SchedulerHelper::AddDataArrow(stack_actor, to_actor, from_index, to_index);
    }
  }
}

void ControlNodeScheduler::LinkArrowbyFormalParameter(ControlActor *const to_actor,
                                                      const KernelWithIndex &from_node_with_index,
                                                      const KernelWithIndex &to_node_with_index,
                                                      const GraphCompilerInfo &graph_compiler_info) const {
  const auto &real_from_node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(from_node_with_index);
  const auto &from_node = real_from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Link arrow by formal parameter, from node:" << from_node->DebugString()
                << " from index:" << real_from_node_with_index.second << " to actor:" << to_actor->GetAID()
                << " to index:" << to_node_with_index.second;
  if (from_node->isa<ValueNode>()) {
    LinkArrowByValueNode(from_node, to_actor, real_from_node_with_index.second, to_node_with_index.second);
  } else if (from_node->isa<Parameter>()) {
    LinkArrowByParameter(from_node, to_actor, real_from_node_with_index, to_node_with_index,
                         graph_compiler_info.control_node_parser_);
  } else if (common::AnfAlgo::IsCallNode(from_node) && !IsNotCut(from_node)) {
    // Link arrow by call node.
    LinkArrowByCallNode(from_node, to_actor, real_from_node_with_index, to_node_with_index, graph_compiler_info);
  } else if (common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitch) ||
             common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimSwitchLayer)) {
    // Link arrow from switch actor.
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (IsPartialInput(from_node)) {
      SchedulerHelper::AddPartialArrow(switch_actor, to_actor, real_from_node_with_index.second,
                                       to_node_with_index.second);
    } else {
      SchedulerHelper::AddDataArrow(switch_actor, to_actor, real_from_node_with_index.second,
                                    to_node_with_index.second);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(from_node, prim::kPrimPartial)) {
    // If the funcgraph of the partial node is a deadnode, in order to ensure the correspondence between formal
    // parameters and real parameters, we need to create an empty partial for it.
    if (IsInvalidPartial(from_node)) {
      MS_LOG(DEBUG) << "Invalid partial node:" << from_node->DebugString();
      to_actor->local_partials_[to_node_with_index.second] = std::make_shared<OpPartial>();
      return;
    }
    // Link arrow from gather actor
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    if (actor == nullptr) {
      MS_LOG(DEBUG) << "No actor of " << actor_name;
      return;
    }
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    SchedulerHelper::AddPartialArrow(gather_actor, to_actor, real_from_node_with_index.second,
                                     to_node_with_index.second);
  } else if (from_node->isa<CNode>()) {
    // Link arrow by kernel.
    LinkArrowByKernel(from_node, to_actor, real_from_node_with_index, to_node_with_index, graph_compiler_info);
  }
}

void ControlNodeScheduler::LinkArrowByValueNode(const AnfNodePtr &value_node, ControlActor *const to_actor,
                                                size_t from_index, size_t to_index) const {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(to_actor);

  if (IsValueNode<FuncGraph>(value_node)) {
    // Link local partial.
    const auto &func_graph = GetValueNode<FuncGraphPtr>(value_node);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Add local partial, graph:" << func_graph->ToString() << " for actor:" << to_actor->GetAID();
    to_actor->local_partials_[to_index] = std::make_shared<OpPartial>();
    *(to_actor->local_partials_[to_index]) = {func_graph.get(), {}, {}};
  } else {
    // Link device store value node.
    if (!AnfAlgo::OutputAddrExist(value_node, from_index)) {
      auto node = value_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      auto value = node->value();
      MS_EXCEPTION_IF_NULL(value);
      // If the from index exceeds the size of the value node, we need to change the from index to 0.
      if (!value->isa<ValueTuple>() && from_index > 0) {
        from_index = 0;
      } else {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, value_node)
          << "#dmsg#Runtime error info:#dmsg#Invalid output address index:" << from_index
          << " for value node:" << value_node->DebugString() << " to actor:" << to_actor->GetAID();
      }
    }
    to_actor->local_kernel_tensors_[to_index] = {AnfAlgo::GetOutputKernelTensor(value_node, from_index, false),
                                                 value_node};
    to_actor->local_kernel_tensors_[to_index].first->device_address()->SetNodeIndex(value_node, from_index);
    MS_LOG(DEBUG) << "Add local device tensor:" << to_actor->local_kernel_tensors_[to_index].first
                  << " index:" << to_index << " for actor:" << to_actor->GetAID() << " from index:" << from_index;
  }
}

void ControlNodeScheduler::LinkArrowByParameter(const AnfNodePtr &parameter, ControlActor *const to_actor,
                                                const KernelWithIndex &from_node_with_index,
                                                const KernelWithIndex &to_node_with_index,
                                                const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(parser);

  MS_LOG(DEBUG) << "Link arrow by parameter:" << parameter->DebugString() << " indx:" << from_node_with_index.second
                << " for actor:" << to_actor->GetAID();

  if (parser->IsRootGraphPersistentDeviceTensor(parameter)) {
    (void)to_actor->device_tensor_store_keys_.emplace_back(to_node_with_index.second, parameter);
    return;
  }

  // Link arrow from entrance actor.
  const auto &func_graph = parameter->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
  auto actor = FetchActor(actor_name);
  MS_EXCEPTION_IF_NULL(actor);

  // If the input of the exit actor of the kernel graph is a parameter node, and there is a corresponding stack actor,
  // it should be linked to the stack actor.
  if (to_actor->type() == KernelTransformType::kExitActor) {
    auto stack_actor_name = (to_actor->node_ == nullptr ? GetStackActorNameByExitName(to_actor->GetAID().Name())
                                                        : GetActorName(to_actor->node_) + kStackActorNameSuffix);
    auto stack_actor = FetchActor(stack_actor_name);
    actor = (stack_actor == nullptr ? actor : stack_actor);
  }

  auto from_actor = dynamic_cast<ControlActor *>(actor);
  MS_EXCEPTION_IF_NULL(from_actor);

  auto abstract = parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto dst_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, from_node_with_index.second);
  MS_EXCEPTION_IF_NULL(dst_abstract);
  if (dst_abstract->isa<abstract::AbstractFunction>()) {
    SchedulerHelper::AddPartialArrow(from_actor, to_actor, from_actor->FetchNodePosition(from_node_with_index),
                                     to_node_with_index.second);
  } else {
    SchedulerHelper::AddDataArrow(from_actor, to_actor, from_actor->FetchNodePosition(from_node_with_index),
                                  to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkArrowByCallNode(const AnfNodePtr &call_node, ControlActor *const to_actor,
                                               const KernelWithIndex &from_node_with_index,
                                               const KernelWithIndex &to_node_with_index,
                                               const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  auto parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  if (to_actor->type_ != KernelTransformType::kEntranceActor) {
    // Link arrow from exit actor to control actor.
    const auto &abstract = call_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, from_node_with_index.second);
    MS_EXCEPTION_IF_NULL(real_abstract);

    std::set<FuncGraphPtr> func_graphs;
    try {
      func_graphs = parser->FetchFuncGraphbyCallNode(from_node);
    } catch (std::exception &e) {
      LinkArrowByKernel(call_node, to_actor, from_node_with_index, to_node_with_index, graph_compiler_info);
      func_graphs.clear();
    }

    for (const auto &func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kExitActorNameSuffix;
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto exit_actor = dynamic_cast<ExitActor *>(actor);
      MS_EXCEPTION_IF_NULL(exit_actor);
      auto branch_id = parser->FetchBranchIDByCallNode(from_node);
      if (real_abstract->isa<abstract::AbstractFunction>()) {
        SchedulerHelper::AddPartialArrowForExitActor(exit_actor, to_actor, from_node_with_index.second,
                                                     to_node_with_index.second, branch_id);
      } else {
        SchedulerHelper::AddDataArrowForExitActor(exit_actor, to_actor, from_node_with_index.second,
                                                  to_node_with_index.second, branch_id);
      }
      MS_LOG(DEBUG) << "Link data arrow from:" << exit_actor->GetAID() << " index:" << from_node_with_index.second
                    << " to:" << to_actor->GetAID() << " index" << to_node_with_index.second;
    }
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      to_actor->input_partials_num_++;
    } else {
      MS_LOG(DEBUG) << "Actor:" << to_actor->GetAID() << " add input num:" << to_actor->input_datas_num_;
      to_actor->input_datas_num_++;
    }
  } else {
    // Link arrow from gather actor to entrance actor.
    const auto &actor_name = GetActorName(from_node);
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    const auto &to_node = to_node_with_index.first;
    MS_EXCEPTION_IF_NULL(to_node);
    const auto &func_graph = to_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    SchedulerHelper::AddDataWithBranchIDArrow(gather_actor, dynamic_cast<EntranceActor *>(to_actor), func_graph);
  }
}

void ControlNodeScheduler::LinkArrowByKernel(const AnfNodePtr &kernel, ControlActor *const to_actor,
                                             const KernelWithIndex &from_node_with_index,
                                             const KernelWithIndex &to_node_with_index,
                                             const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &graph = parser->FetchKernelGraphByFrontNode(from_node);
  MS_LOG(DEBUG) << "Link arrow by kernel, from mode:" << from_node->DebugString() << " to actor:" << to_actor->GetAID();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &group_name = parser->FetchGroupNameByKernelGraph(graph);

  if (to_actor->type_ == KernelTransformType::kExitActor && to_actor->node_ == nullptr &&
      to_actor->GetAID().Name().find(group_name) != std::string::npos) {
    // Link arrow from actor of output node to exit actor of kernel graph.
    auto kernel_with_index = parser->FetchBackendNodeByFrontNode(from_node_with_index);
    auto backoff_kernel_with_index = parser->FetchBackendOutputByKernelGraph(graph, from_node_with_index);
    // If front node and backend node are not the same type, maybe the output node has been replaced by pass,
    // the output arrow should be linked to the new node.
    if (backoff_kernel_with_index.first != nullptr && backoff_kernel_with_index.first->isa<ValueNode>()) {
      MS_LOG(INFO) << "Backend node has been replaced to:" << backoff_kernel_with_index.first->DebugString();
      LinkArrowByValueNode(backoff_kernel_with_index.first, to_actor, backoff_kernel_with_index.second,
                           to_node_with_index.second);
      return;
    }
    if (kernel_with_index.first == nullptr) {
      kernel_with_index = backoff_kernel_with_index;
      if (kernel_with_index.first == nullptr) {
        parser->PrintParseInfo();
        MS_LOG_WITH_NODE(EXCEPTION, from_node)
          << "Failed to get kernel with index by front node:" << from_node->fullname_with_scope()
          << " debug string:" << from_node->DebugString() << " index:" << from_node_with_index.second
          << " by graph:" << graph->ToString() << " to actor:" << to_actor->GetAID();
      }
    }
    auto type = FetchKernelTransformType(kernel_with_index.first, graph, {});
    auto from_actor = FetchActor(type, graph_compiler_info.name_, kernel_with_index.first, graph);
    if (from_actor == nullptr) {
      parser->PrintParseInfo();
      MS_LOG_WITH_NODE(EXCEPTION, from_node)
        << "Failed to get from actor by backend node:" << kernel_with_index.first->DebugString()
        << " front node : " << from_node->fullname_with_scope() << " debug string:" << from_node->DebugString()
        << " index:" << from_node_with_index.second << " by graph:" << graph->ToString()
        << " to actor:" << to_actor->GetAID() << " type:" << type;
    }
    SchedulerHelper::AddDataArrow(from_actor, to_actor, kernel_with_index.second, to_node_with_index.second,
                                  kernel_with_index.first);
  } else {
    // Link arrow from exit actor of kernel graph to exit actor of function graph.
    const auto &actor_name = parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix;
    MS_LOG(DEBUG) << "Actor name:" << actor_name << " from node:" << from_node->DebugString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto exit_actor = dynamic_cast<ExitActor *>(actor);
    MS_EXCEPTION_IF_NULL(exit_actor);
    size_t from_index = exit_actor->FetchNodePosition(from_node_with_index);
    SchedulerHelper::AddDataArrow(exit_actor, to_actor, from_index, to_node_with_index.second);
  }
}

void ControlNodeScheduler::LinkControlArrowForControlActor(ActorSet *const actor_set,
                                                           const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto control_actor_set = actor_set->control_actors_.get();
  MS_EXCEPTION_IF_NULL(control_actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  LinkControlArrowForEntranceActor(actor_set, graph_compiler_info);

  // When the switch actor and gather actor have no input, need to link a control arrow from entrance actor.
  std::vector<ControlActor *> need_check_control_actors;
  (void)std::transform(control_actor_set->switch_actors_.begin(), control_actor_set->switch_actors_.end(),
                       std::back_inserter(need_check_control_actors),
                       [](const auto &switch_actor) { return switch_actor.get(); });
  (void)std::transform(control_actor_set->gather_actors_.begin(), control_actor_set->gather_actors_.end(),
                       std::back_inserter(need_check_control_actors),
                       [](const auto &gather_actor) { return gather_actor.get(); });

  for (auto control_actor : need_check_control_actors) {
    MS_EXCEPTION_IF_NULL(control_actor);
    if (IsNoInputActor(control_actor)) {
      MS_EXCEPTION_IF_NULL(control_actor->node_);
      if (parser->IsNeedStackControlNode(control_actor->node_)) {
        const auto &stack_actor_name = GetActorName(control_actor->node_) + kStackActorNameSuffix;
        auto actor = FetchActor(stack_actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        auto to_actor = dynamic_cast<ControlActor *>(actor);
        MS_EXCEPTION_IF_NULL(to_actor);
        SchedulerHelper::AddControlArrow(to_actor, control_actor);
        continue;
      }
      const FuncGraphPtr &func_graph = control_actor->node_->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
      const auto &entrance_actor = dynamic_cast<EntranceActor *>(FetchActor(actor_name));
      MS_EXCEPTION_IF_NULL(entrance_actor);
      SchedulerHelper::AddControlArrow(entrance_actor, control_actor);
    }
  }

  // Link auto monad control arrow for control actor.
  std::vector<ControlActor *> control_actors;
  (void)std::transform(control_actor_set->switch_actors_.begin(), control_actor_set->switch_actors_.end(),
                       std::back_inserter(control_actors), [](auto &switch_actor) { return switch_actor.get(); });
  (void)std::transform(control_actor_set->gather_actors_.begin(), control_actor_set->gather_actors_.end(),
                       std::back_inserter(control_actors), [](auto &gather_actor) { return gather_actor.get(); });
  (void)std::transform(control_actor_set->exit_actors_.begin(), control_actor_set->exit_actors_.end(),
                       std::back_inserter(control_actors), [](auto &exit_actor) { return exit_actor.get(); });
  for (auto control_actor : control_actors) {
    MS_EXCEPTION_IF_NULL(control_actor);
    const auto &node = control_actor->node_;
    if (node == nullptr) {
      continue;
    }

    auto to_actor = control_actor;
    if (parser->IsNeedStackControlNode(node)) {
      const auto &stack_actor_name = GetActorName(node) + kStackActorNameSuffix;
      auto actor = FetchActor(stack_actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      to_actor = dynamic_cast<ControlActor *>(actor);
      MS_EXCEPTION_IF_NULL(to_actor);
    }

    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      std::vector<AnfNodePtr> monad_nodes = FetchAllMonadNodeByNode(input);
      for (const auto &monad_node : monad_nodes) {
        MS_EXCEPTION_IF_NULL(monad_node);
        LinkControlArrowByAutoMonad(to_actor, monad_node, parser);
      }
    }
  }

  LinkControlArrowByKernelGraphGroup(graph_compiler_info);
}

void ControlNodeScheduler::LinkControlArrowForEntranceActor(ActorSet *const actor_set,
                                                            const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto control_actor_set = actor_set->control_actors_.get();
  MS_EXCEPTION_IF_NULL(control_actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Since only one set of real parameters are allowed to be executed in funcgraph at the same time, when the funcgraph
  // stops running, it is necessary to send the control arrow to the corresponding entrance actor at the exit of the
  // graph to run the next set of real parameters. The corresponding nodes of the actors that need to send the control
  // arrow have been parsed in the control node parser.
  for (const auto &graph_to_nodes : parser->func_graph_to_first_control_nodes_) {
    // Fetch the entrance actor.
    const auto &func_graph = graph_to_nodes.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto entrance_actor = dynamic_cast<EntranceActor *>(FetchActor(actor_name));
    MS_EXCEPTION_IF_NULL(entrance_actor);

    const auto &nodes = graph_to_nodes.second;
    for (const auto &node : nodes) {
      // Fetch the source actor of control arrow.
      MS_EXCEPTION_IF_NULL(node);
      if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
        actor_name = func_graph->ToString() + kExitActorNameSuffix;
      } else {
        actor_name = GetActorName(node);
      }
      auto from_actor = dynamic_cast<ControlActor *>(FetchActor(actor_name));
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddLoopBodyControlArrow(from_actor, entrance_actor);
    }
  }

  // In the recursive scene, some kernel graph needs to be completed before the next set of data is sent by the
  // entrance actor. At this time, it is necessary to connect a control arrow from the exit actor of the graph
  // to the entrance actor.
  for (const auto &func_graph_to_group_info : parser->func_graph_to_first_kernel_graphs_) {
    const auto &func_graph = func_graph_to_group_info.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    for (const auto &group_info : func_graph_to_group_info.second) {
      MS_EXCEPTION_IF_NULL(group_info);
      actor_name = group_info->group_name_ + kExitActorNameSuffix;
      auto from_actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddLoopBodyControlArrow(from_actor, entrance_actor);
    }
  }
}

void ControlNodeScheduler::LinkControlArrowForLoopCountActor(const ActorSet *actor_set,
                                                             const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto loop_count_actor = actor_set->loop_count_actor_;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  // The final output is always sent by the exit of the root graph in control flow.
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &root_graph = parser->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  auto exit_actor_name = root_graph->ToString() + kExitActorNameSuffix;
  auto root_exit_actor = dynamic_cast<ExitActor *>(FetchActor(exit_actor_name));
  MS_EXCEPTION_IF_NULL(root_exit_actor);
  // link control arrow from root exit actor to loop count actor.
  SchedulerHelper::AddControlArrowForExitActor(root_exit_actor, loop_count_actor.get(), kMainBranchID);

  // The entrance actor will generate some data in the loop body execution, so need clear on the end of step.
  MS_EXCEPTION_IF_NULL(actor_set->control_actors_);
  for (auto &entrance_actor : actor_set->control_actors_->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    (void)loop_count_actor->entrance_aids_.emplace_back(entrance_actor->GetAID());
  }
}

void ControlNodeScheduler::LinkOutputControlArrowForActor(ActorSet *const actor_set,
                                                          const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  // Link control arrows from no super kernel actor to the corresponding exit actor.
  for (auto &super_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_actor);
    if ((super_actor->output_data_arrows_.size() == 0) && (super_actor->output_control_arrows_.size() == 0)) {
      auto kernel_graph = super_actor->graph();
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto to_actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kExitActorNameSuffix;
      auto to_actor = FetchActor(to_actor_name);
      MS_EXCEPTION_IF_NULL(to_actor);
      SchedulerHelper::AddControlArrow(super_actor.get(), to_actor);
    }
  }
}

void ControlNodeScheduler::LinkControlArrowForKernelActor(ActorSet *const actor_set,
                                                          const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link control arrow from entrance actors or stack actors to no input kernel actors.
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    // In control flow, when the input of the kernel actor is a parameter, this input needs to be linked to the
    // control actor, so the no-input kernel actor collected in the graph scheduler will also collect this actor,
    // and it needs to be skipped here.
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    // Control arrow for custom actor will be linked in next step.
    if ((no_input_kernel_actor->input_datas_num_ != 0) || (no_input_kernel_actor->input_controls_num_ != 0)) {
      continue;
    }

    KernelGraphPtr kernel_graph = nullptr;
    if (no_input_kernel_actor->type_ == KernelTransformType::kSuperKernelActor) {
      const auto &super_kernel_actor = dynamic_cast<SuperKernelActor *>(no_input_kernel_actor.get());
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      kernel_graph = super_kernel_actor->graph();
    } else {
      MS_LOG(EXCEPTION) << "Invalid no input actor: " << no_input_kernel_actor->GetAID().Name();
    }
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto actor_name = parser->FetchGroupNameByKernelGraph(kernel_graph) + kStackActorNameSuffix;
    if (!parser->IsCallInputKernelGraph(kernel_graph.get())) {
      const auto &func_graph = parser->FetchFuncGraphByKernelGraph(kernel_graph.get());
      MS_EXCEPTION_IF_NULL(func_graph);
      actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    }

    auto from_actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(from_actor);
    SchedulerHelper::AddControlArrow(from_actor, no_input_kernel_actor.get());
  }
  LinkOutputControlArrowForActor(actor_set, graph_compiler_info);
}

void ControlNodeScheduler::LinkControlArrowByAutoMonad(ControlActor *to_actor, const AnfNodePtr &from_node,
                                                       const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link auto monad control arrow from node:" << from_node->DebugString()
                << " to actor:" << to_actor->GetAID();

  std::set<AnfNodePtr> depend_nodes;
  FetchRealDependNodeByAutoMonad(from_node, &depend_nodes);

  for (const auto &depend_node : depend_nodes) {
    MS_EXCEPTION_IF_NULL(depend_node);
    MS_LOG(DEBUG) << "Add depend node:" << depend_node->DebugString() << " for actor:" << to_actor->GetAID();
    auto from_actor = FetchActor(GetActorName(depend_node));
    auto graph = parser->FetchKernelGraphByFrontNode(depend_node);

    std::vector<AbstractActor *> from_actors;
    if (common::AnfAlgo::IsCallNode(depend_node) && !IsNotCut(depend_node)) {
      // If the actor already exists with control arrow, skip it.
      if (IsControlArrowExistForCallNode(depend_node, to_actor, parser)) {
        MS_LOG(DEBUG) << "Control arrow from call node:" << depend_node << " to actor:" << to_actor->GetAID()
                      << "is exist, skip it";
        continue;
      }
      int branch_id = parser->FetchBranchIDByCallNode(depend_node);
      const auto &func_graphs = parser->FetchFuncGraphbyCallNode(depend_node);
      if (func_graphs.empty()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, depend_node)
          << "#dmsg#Runtime error info:#dmsg#Failed to get funcgraph by call node:" << depend_node->DebugString();
      }
      for (const auto &func_graph : func_graphs) {
        MS_EXCEPTION_IF_NULL(func_graph);
        auto exit_actor_name = func_graph->ToString() + kExitActorNameSuffix;
        from_actor = FetchActor(exit_actor_name);
        MS_EXCEPTION_IF_NULL(from_actor);
        (void)from_actors.emplace_back(from_actor);
        auto exit_actor = dynamic_cast<ExitActor *>(from_actor);
        MS_EXCEPTION_IF_NULL(exit_actor);
        SchedulerHelper::AddControlArrowForExitActor(exit_actor, to_actor, branch_id);
      }
      to_actor->input_controls_num_ -= (func_graphs.size() - 1);
    } else if (from_actor != nullptr) {
      (void)from_actors.emplace_back(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    } else {
      if (graph == nullptr) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, depend_node)
          << "#dmsg#Runtime error info:#dmsg#Failed to find actor for node:" << depend_node->DebugString();
      }
      from_actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kExitActorNameSuffix);
      MS_EXCEPTION_IF_NULL(from_actor);
      if (std::find_if(from_actor->output_control_arrows_.begin(), from_actor->output_control_arrows_.end(),
                       [&to_actor](auto &output_control_arrow) {
                         MS_EXCEPTION_IF_NULL(output_control_arrow);
                         return output_control_arrow->to_op_id_.Name() == to_actor->GetAID().Name();
                       }) != from_actor->output_control_arrows_.end()) {
        MS_LOG(DEBUG) << "Link auto monad control from actor:" << from_actor->GetAID()
                      << " to actor:" << to_actor->GetAID() << " is already exist.";
        continue;
      }
      (void)from_actors.emplace_back(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    }
    if (to_actor->type_ != KernelTransformType::kStackActor || parser->IsNeedStackControlNode(depend_node) ||
        parser->IsRecursionCallNode(depend_node) || (graph != nullptr && parser->IsRecursionKernelGraph(graph))) {
      continue;
    }
    // If the control arrow comes from a recursive call node or a recursive kernel graph, these control edges will be
    // directly linked to the stack actor, otherwise, they need to be cached in the stack of the stack actor.
    auto stack_actor = dynamic_cast<StackActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(stack_actor);
    stack_actor->input_controls_num_--;
    stack_actor->input_stack_controls_num_++;
    for (const auto &actor : from_actors) {
      MS_EXCEPTION_IF_NULL(actor);
      MS_LOG(DEBUG) << "Add stack control aid:" << actor->GetAID() << " for actor:" << stack_actor->GetAID();
      (void)stack_actor->stack_control_aids_.emplace(actor->GetAID());
      stack_actor->control_aid_to_indexs_[actor->GetAID()] = stack_actor->input_stack_controls_num_;
    }
  }
  MS_LOG(DEBUG) << "Link auto monad control arrow from node:" << from_node->DebugString()
                << " to actor:" << to_actor->GetAID() << " end";
}

void ControlNodeScheduler::LinkControlArrowByKernelGraphGroup(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &graph_group : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(graph_group);
    if (!graph_group->need_stack_) {
      continue;
    }
    auto stack_actor = FetchActor(graph_group->group_name_ + kStackActorNameSuffix);
    MS_EXCEPTION_IF_NULL(stack_actor);
    auto to_actor = dynamic_cast<ControlActor *>(stack_actor);
    MS_EXCEPTION_IF_NULL(to_actor);
    for (const auto &monad_input : graph_group->monad_inputs_) {
      MS_EXCEPTION_IF_NULL(monad_input);
      MS_LOG(DEBUG) << "Add monad control arrow for group:" << graph_group->group_name_
                    << " to actor:" << to_actor->GetAID() << " by monad input:" << monad_input->DebugString();
      LinkControlArrowByAutoMonad(to_actor, monad_input, parser);
    }
  }
}

void ControlNodeScheduler::LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set) const {
  MS_EXCEPTION_IF_NULL(control_actor_set);

  // Connect the branch id arrows from the entrance actor to the exit actor for each funcgraph.
  for (auto exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);

    // If the node in the exit actor is empty, it means that it is between the kernel actor and the control actor,
    // and no need to send the branch id.
    const auto &node = exit_actor->node_;
    if (node == nullptr) {
      continue;
    }

    const auto &func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    SchedulerHelper::AddBranchIDArrow(entrance_actor, exit_actor.get());
  }

  // Connect the branch id arrows from the entrance actor to the stack actor.
  for (auto stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    auto node = stack_actor->node_;
    if (!stack_actor->formal_parameters_.empty()) {
      node = stack_actor->formal_parameters_.back().first;
    } else {
      MS_LOG(INFO) << "No formal parameter for stack actor:" << stack_actor->GetAID();
    }
    MS_EXCEPTION_IF_NULL(node);
    const auto &func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    SchedulerHelper::AddBranchIDArrow(entrance_actor, stack_actor.get());
  }
}

void ControlNodeScheduler::LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Link data arrows from entrance actors and stack actors to kernel actors.
  for (const auto &func_graph_to_kernel_graphs : parser->func_graph_to_kernel_graph_groups_) {
    // Fetch the source entrance actor.
    const auto &func_graph = func_graph_to_kernel_graphs.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto entrance_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);

    for (const auto &kernel_graph_group : func_graph_to_kernel_graphs.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_EXCEPTION_IF_NULL(kernel_graph);
        if (kernel_graph->execution_order().empty()) {
          continue;
        }
        LinkDataArrowByKernelGraph(kernel_graph, entrance_actor, parser);
      }
    }
  }
}

void ControlNodeScheduler::LinkDataArrowByKernelGraphInSinkMode(const KernelGraphPtr &graph,
                                                                ControlActor *const from_actor,
                                                                const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(parser);
  MS_LOG(DEBUG) << "Link data arrow in sink mode by kernel graph:" << graph->ToString();
  auto to_actor = FetchActor(KernelTransformType::kSuperKernelActor, "", nullptr, graph);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto super_kernel_actor = dynamic_cast<SuperKernelActor *>(to_actor);
  MS_EXCEPTION_IF_NULL(super_kernel_actor);

  auto &input_nodes = graph->input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    const auto &input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node) || (!parser->IsControlFlowDataArrow(graph, input_node))) {
      continue;
    }
    size_t to_index = super_kernel_actor->FetchInputNodePosition(input_node);
    const auto &front_node_with_index = GetFrontNodeByKernelGraph(input_node, graph.get());
    MS_EXCEPTION_IF_NULL(front_node_with_index.first);
    if (front_node_with_index.first->isa<ValueNode>()) {
      continue;
    }
    if (front_node_with_index.first->isa<CNode>() && (from_actor->type() != KernelTransformType::kStackActor)) {
      // If the input is an internal parameter, the input arrow should be linked to the exit actor of the kernel
      // graph which the internal parameter belong.
      MS_LOG(INFO) << "Internal parameter in control flow, backend input:" << input_node->DebugString()
                   << " front node:" << front_node_with_index.first->DebugString();
      const auto &from_graph = parser->FetchKernelGraphByFrontNode(front_node_with_index.first);
      MS_EXCEPTION_IF_NULL(from_graph);
      auto actor = FetchActor(parser->FetchGroupNameByKernelGraph(from_graph) + kExitActorNameSuffix);
      MS_EXCEPTION_IF_NULL(actor);
      auto exit_actor = dynamic_cast<ControlActor *>(actor);
      MS_EXCEPTION_IF_NULL(exit_actor);
      size_t from_index = exit_actor->FetchNodePosition(front_node_with_index);
      SchedulerHelper::AddDataArrow(exit_actor, to_actor, from_index, i);
      continue;
    }
    size_t from_index = from_actor->FetchNodePosition(front_node_with_index);
    SchedulerHelper::AddDataArrow(from_actor, to_actor, from_index, to_index);
  }
  return;
}

void ControlNodeScheduler::LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, ControlActor *const entrance_actor,
                                                      const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(entrance_actor);
  MS_LOG(DEBUG) << "Link data arrow by kernel graph:" << graph->ToString();
  auto from_actor = entrance_actor;
  // If there is a call node in the input of the graph, the parameter of the graph needs to be sent by the
  // corresponding stack actor, otherwise it is sent by the entrance actor.
  if (parser->IsCallInputKernelGraph(graph.get())) {
    auto actor = FetchActor(parser->FetchGroupNameByKernelGraph(graph) + kStackActorNameSuffix);
    MS_EXCEPTION_IF_NULL(actor);
    from_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(from_actor);
  }

  // Link data arrow in graph mode.
  LinkDataArrowByKernelGraphInSinkMode(graph, from_actor, parser);
}

void ControlNodeScheduler::LinkDataArrowForOutputActor(ActorSet *const actor_set,
                                                       const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto &to_actor = actor_set->output_actor_;
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  const auto &root_graph = parser->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &return_node = root_graph->return_node();
  MS_EXCEPTION_IF_NULL(return_node);

  const auto &exit_actor_name = root_graph->ToString() + kExitActorNameSuffix;
  auto actor = FetchActor(exit_actor_name);
  MS_EXCEPTION_IF_NULL(actor);
  auto exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);
  for (size_t i = 0; i < exit_actor->formal_parameters_.size(); ++i) {
    SchedulerHelper::AddDataArrowForExitActor(exit_actor, to_actor.get(), i, i, 0);
    to_actor->input_datas_num_++;
  }
}

void ControlNodeScheduler::LinkArrowForRootGraphEntranceActor(const ActorSet *actor_set,
                                                              const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  const auto &root_graph = graph_compiler_info.control_node_parser_->root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &entrance_actor_name = root_graph->ToString() + kEntranceActorNameSuffix;
  auto to_actor = dynamic_cast<EntranceActor *>(FetchActor(entrance_actor_name));
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &host_ds_actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
  auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(host_ds_actor_name));
  // No host data source actor scenario.
  if (host_ds_actor == nullptr) {
    const auto &data_prepare_actor_name = graph_compiler_info.name_ + kDataPrepareActorNameSuffix;
    auto data_prepare_actor = FetchActor(data_prepare_actor_name);
    MS_EXCEPTION_IF_NULL(data_prepare_actor);
    SchedulerHelper::AddControlArrow(data_prepare_actor, to_actor);
    return;
  }

  // The host data source actor sends all the input to the entrance actor of the root graph.
  for (size_t i = 0; i < to_actor->formal_parameters_.size(); ++i) {
    const auto &formal_parameter = to_actor->formal_parameters_[i];
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    MS_LOG(DEBUG) << "Formal parameter:" << formal_parameter.first->DebugString()
                  << " index:" << formal_parameter.second;
    const auto &iter = host_ds_actor->data_node_position_map_.find(formal_parameter);
    if (iter != host_ds_actor->data_node_position_map_.end()) {
      const auto &parameter_with_index = host_ds_actor->data_nodes()[iter->second];
      SchedulerHelper::AddDataArrow(host_ds_actor, to_actor, parameter_with_index.second, i,
                                    parameter_with_index.first);
    } else {
      MS_LOG(INFO) << "Invalid formal parameter:" << formal_parameter.first->DebugString()
                   << " index:" << formal_parameter.second << " for actor:" << to_actor->GetAID();
    }
  }
}

void ControlNodeScheduler::SetTimeSummaryForControlActor(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &kernel_graph_group_info : parser->kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    const auto &exit_actor_name = kernel_graph_group_info->group_name_ + kExitActorNameSuffix;
    const auto &exit_base_actor = FetchActor(exit_actor_name);
    if (exit_base_actor == nullptr) {
      continue;
    }
    const auto &exit_actor = dynamic_cast<ControlActor *>(exit_base_actor);
    MS_EXCEPTION_IF_NULL(exit_actor);

    // Set the exit actor of kernel graph to its entrance actor or stack actor.
    if (kernel_graph_group_info->need_stack_ == false) {
      if (kernel_graph_group_info->graphs_.empty()) {
        continue;
      }
      const auto &graph = *(kernel_graph_group_info->graphs_.begin());
      const auto &func_graph = parser->FetchFuncGraphByKernelGraph(graph.get());
      MS_EXCEPTION_IF_NULL(func_graph);
      auto entrance_base_actor = FetchActor(func_graph->ToString() + kEntranceActorNameSuffix);
      if (entrance_base_actor != nullptr) {
        const auto &entrance_actor = dynamic_cast<ControlActor *>(entrance_base_actor);
        MS_EXCEPTION_IF_NULL(entrance_actor);
        (void)entrance_actor->end_actors_.emplace(exit_actor);
        MS_LOG(DEBUG) << "Add time summart for exit actor:" << exit_actor->GetAID()
                      << " to actor:" << entrance_actor->GetAID();
      }
      continue;
    }

    auto stack_base_actor = FetchActor(kernel_graph_group_info->group_name_ + kStackActorNameSuffix);
    if (stack_base_actor != nullptr) {
      const auto &stack_actor = dynamic_cast<ControlActor *>(stack_base_actor);
      MS_EXCEPTION_IF_NULL(stack_actor);
      (void)stack_actor->end_actors_.emplace(exit_actor);
      MS_LOG(DEBUG) << "Add time summart for exit actor:" << exit_actor->GetAID()
                    << " to actor:" << stack_actor->GetAID();
    }
  }
}

bool ControlNodeScheduler::IsNoInputActor(const ControlActor *control_actor) const {
  MS_EXCEPTION_IF_NULL(control_actor);
  return (control_actor->input_datas_num_ == 0 && control_actor->input_controls_num_ == 0 &&
          control_actor->input_partials_num_ == 0 && control_actor->input_branch_ids_num_ == 0);
}

std::vector<std::string> ControlNodeScheduler::GetInputAids(
  AbstractActor *const actor, const ControlNodeParserPtr &parser,
  const std::unordered_map<std::string, std::string> &exit_to_gather, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!IsControlFlowActor(actor->type())) {
    MS_LOG(EXCEPTION) << "Invalid actor:" << actor->GetAID() << " for control actor topo sort.";
  }
  if (actor->type() == KernelTransformType::kEntranceActor) {
    return {};
  }
  const auto &control_actor = dynamic_cast<ControlActor *>(actor);
  MS_EXCEPTION_IF_NULL(control_actor);
  std::vector<std::string> input_aids;

  auto get_relative_aid = [&exit_to_gather, &control_actor, &func_graph](const std::string &aid) {
    const auto &actor = FetchActor(aid);
    MS_EXCEPTION_IF_NULL(actor);
    MS_EXCEPTION_IF_NULL(func_graph);
    if (aid.find(kExitActorNameSuffix) != std::string::npos) {
      const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
      MS_EXCEPTION_IF_NULL(exit_actor);
      if (exit_actor->node() != nullptr) {
        auto key = exit_actor->GetAID().Name() + control_actor->GetAID().Name();
        const auto &iter = exit_to_gather.find(key);
        if (iter == exit_to_gather.end()) {
          MS_LOG(EXCEPTION) << "Failed to get gather actor by from actor:" << exit_actor->GetAID()
                            << " to:" << control_actor->GetAID();
        }
        return iter->second;
      }
    }
    if (!IsControlFlowActor(actor->type())) {
      if (control_actor->type() != KernelTransformType::kExitActor) {
        MS_LOG(EXCEPTION) << "Invalid input actor:" << actor->GetAID() << " for actor:" << control_actor->GetAID();
      }
      const auto &stack_actor_name = GetStackActorNameByExitName(control_actor->GetAID().Name());
      const auto &stack_actor = FetchActor(stack_actor_name);
      if (stack_actor == nullptr) {
        return func_graph->ToString() + kEntranceActorNameSuffix;
      }
      return stack_actor_name;
    }
    return aid;
  };

  std::for_each(control_actor->input_data_arrow_aids().begin(), control_actor->input_data_arrow_aids().end(),
                [&input_aids, &get_relative_aid](const std::pair<AID, DataArrow *> &pair) {
                  input_aids.emplace_back(get_relative_aid(pair.first.Name()));
                });
  std::for_each(control_actor->input_control_arrow_aids().begin(), control_actor->input_control_arrow_aids().end(),
                [&input_aids, &get_relative_aid](const std::pair<AID, ControlArrow *> &pair) {
                  input_aids.emplace_back(get_relative_aid(pair.first.Name()));
                });
  std::for_each(control_actor->input_partial_arrow_aids().begin(), control_actor->input_partial_arrow_aids().end(),
                [&input_aids, &get_relative_aid](const std::pair<AID, DataArrow *> &pair) {
                  input_aids.emplace_back(get_relative_aid(pair.first.Name()));
                });
  std::for_each(control_actor->input_branch_id_arrow_aids().begin(), control_actor->input_branch_id_arrow_aids().end(),
                [&input_aids](const AID &aid) { input_aids.emplace_back(aid.Name()); });
  MS_LOG(DEBUG) << "Actor:" << actor->GetAID() << " input aid num:" << input_aids.size();
  for (const auto &aid : input_aids) {
    MS_LOG(DEBUG) << "Input aid:" << aid << " for actor:" << actor->GetAID();
  }
  return input_aids;
}

namespace {
const char kStubActorNameSuffix[] = "_StubActor";
class StubActor : public AbstractActor {
 public:
  explicit StubActor(const std::string &name, KernelTransformType type, const AID *recorder_aid,
                     const std::vector<FuncGraph *> &graphs, const std::string &exit_actor_name)
      : AbstractActor(name, type, recorder_aid), graphs_(graphs), exit_actor_name_(exit_actor_name) {}
  ~StubActor() override = default;
  void AddInputAid(const AID &aid, DataArrow *arrow) {
    input_data_arrow_aids_.emplace_back(std::make_pair(aid, arrow));
  }
  std::vector<DataArrowPtr> arrows_;
  std::vector<FuncGraph *> graphs_;
  std::string exit_actor_name_;
};
std::string GetActorDumpName(AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  switch (actor->type()) {
    case KernelTransformType::kEntranceActor:
      return "EntranceActor";
    case KernelTransformType::kExitActor:
      return "ExitActor";
    case KernelTransformType::kGatherActor:
      return "GatherActor";
    case KernelTransformType::kSwitchActor:
      return "SwitchActor";
    case KernelTransformType::kStackActor:
      return "StackActor";
    case KernelTransformType::kUnknown:
      if (actor->GetAID().Name().find(kStubActorNameSuffix) != std::string::npos) {
        std::string name = "RunGraph[";
        const auto &stub_actor = dynamic_cast<StubActor *>(actor);
        MS_EXCEPTION_IF_NULL(stub_actor);
        for (size_t i = 0; i < stub_actor->graphs_.size(); ++i) {
          const auto &graph = stub_actor->graphs_[i];
          MS_EXCEPTION_IF_NULL(graph);
          name += graph->ToString();
          if (i + 1 < stub_actor->graphs_.size()) {
            name += ", ";
          }
        }
        name += "]";
        return name;
      }
    default:
      return actor->GetAID().Name();
  }
}
struct InputInfo {
  std::string name;
  size_t index;
};

std::string GetValueNodeName(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  if (value_node->value() != nullptr) {
    if (value_node->value()->isa<Scalar>()) {
      return value_node->value()->DumpText();
    }
    return value_node->value()->ToString();
  }
  return value_node->DebugString();
}

void GetNameForStubActor(AbstractActor *const actor, DataArrow *const dst_arrow, std::string *name, size_t *index) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(dst_arrow);
  MS_EXCEPTION_IF_NULL(name);
  MS_EXCEPTION_IF_NULL(index);
  const auto &stub_actor = dynamic_cast<StubActor *>(actor);
  MS_EXCEPTION_IF_NULL(stub_actor);
  const auto &from_actor = FetchActor(*name);
  if (from_actor == nullptr || from_actor->type() != KernelTransformType::kSuperKernelActor ||
      stub_actor->graphs_.empty() || stub_actor->graphs_[0] == nullptr) {
    return;
  }
  const auto &super_kernel_actor = dynamic_cast<SuperKernelActor *>(from_actor);
  MS_EXCEPTION_IF_NULL(super_kernel_actor);
  MS_EXCEPTION_IF_NULL(super_kernel_actor->graph());
  *name = stub_actor->exit_actor_name_ + super_kernel_actor->graph()->ToString() + kStubActorNameSuffix;
  for (size_t i = 0; i < super_kernel_actor->output_data_arrows().size(); ++i) {
    const auto &arrow = super_kernel_actor->output_data_arrows()[i];
    if (arrow == nullptr || arrow.get() != dst_arrow) {
      continue;
    }
    if (i >= super_kernel_actor->output_data_nodes().size()) {
      return;
    }
    const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(super_kernel_actor->graph()->output());
    auto output_pair = std::make_pair(super_kernel_actor->output_data_nodes()[i], IntToSize(arrow->from_output_index_));
    const auto &output_iter = std::find(outputs.begin(), outputs.end(), output_pair);
    if (output_iter != outputs.end()) {
      *index = output_iter - outputs.begin();
    }
    return;
  }
}

void GetNameForExitActor(AbstractActor *const actor, DataArrow *const dst_arrow,
                         const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                        session::KernelWithIndexCmp> &graph_output_to_actor,
                         std::string *name, size_t *index) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(dst_arrow);
  MS_EXCEPTION_IF_NULL(name);
  MS_EXCEPTION_IF_NULL(index);
  const auto &exit_actor = dynamic_cast<ExitActor *>(actor);
  MS_EXCEPTION_IF_NULL(exit_actor);
  if (exit_actor->node() != nullptr) {
    return;
  }
  size_t input_index = IntToSize(dst_arrow->to_input_index_);
  if (input_index >= exit_actor->formal_parameters().size()) {
    MS_LOG(EXCEPTION) << "Invalid input index:" << input_index << " for actor:" << exit_actor->GetAID();
  }
  const auto &front_parameter = exit_actor->formal_parameters()[input_index];
  const auto &iter = graph_output_to_actor.find(front_parameter);
  if (iter == graph_output_to_actor.end() || iter->second.second.first == nullptr) {
    return;
  }
  const auto &graph = iter->second.second.first->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  *name = exit_actor->GetAID().Name() + graph->ToString() + kStubActorNameSuffix;
  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  const auto &output_iter = std::find(outputs.begin(), outputs.end(), iter->second.second);
  if (output_iter != outputs.end()) {
    *index = output_iter - outputs.begin();
  }
}

void GetInputNameForControlActor(AbstractActor *const actor, std::map<size_t, InputInfo> *input_aids,
                                 size_t *max_index) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(input_aids);
  MS_EXCEPTION_IF_NULL(max_index);
  const auto &control_actor = dynamic_cast<ControlActor *>(actor);
  MS_EXCEPTION_IF_NULL(control_actor);
  for (const auto &pair : control_actor->input_partial_arrow_aids()) {
    MS_EXCEPTION_IF_NULL(pair.second);
    size_t input_index = IntToSize(pair.second->to_input_index_);
    (*input_aids)[input_index] = {pair.first.Name(), IntToSize(pair.second->from_output_index_)};
    *max_index = (*max_index > input_index ? *max_index : input_index);
  }
  for (const auto &pair : control_actor->local_partials()) {
    MS_EXCEPTION_IF_NULL(pair.second);
    (*input_aids)[pair.first] = {(pair.second->func_graph_ == nullptr ? "null" : pair.second->func_graph_->ToString()),
                                 0};
    *max_index = (*max_index > pair.first ? *max_index : pair.first);
  }
  for (const auto &pair : control_actor->local_kernel_tensors()) {
    MS_EXCEPTION_IF_NULL(pair.second.second);
    std::string name = pair.second.second->DebugString(0);
    if (pair.second.second->isa<ValueNode>()) {
      name = GetValueNodeName(pair.second.second->cast<ValueNodePtr>());
    }
    (*input_aids)[pair.first] = {name, 0};
    *max_index = (*max_index > pair.first ? *max_index : pair.first);
  }
}

void GetAllInputByArrow(AbstractActor *const actor,
                        const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                       session::KernelWithIndexCmp> &graph_output_to_actor,
                        std::map<size_t, InputInfo> *input_aids, size_t *max_index) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(input_aids);
  MS_EXCEPTION_IF_NULL(max_index);
  for (const auto &pair : actor->input_data_arrow_aids()) {
    MS_EXCEPTION_IF_NULL(pair.second);
    size_t input_index = IntToSize(pair.second->to_input_index_);
    std::string name = pair.first.Name();
    size_t index = IntToSize(pair.second->from_output_index_);
    if (actor->type() == KernelTransformType::kExitActor) {
      GetNameForExitActor(actor, pair.second, graph_output_to_actor, &name, &index);
    } else if (actor->type() == KernelTransformType::kUnknown) {
      const auto &input_actor = FetchActor(name);
      if (input_actor == nullptr) {
        MS_LOG(WARNING) << "Failed to get actor by name:" << name;
        continue;
      }

      GetNameForStubActor(actor, pair.second, &name, &index);
    }
    (*input_aids)[input_index] = {name, index};
    *max_index = (*max_index > input_index ? *max_index : input_index);
  }
}

// Get string of all input actor.
std::map<size_t, InputInfo> GetInputName(AbstractActor *const actor, const ControlNodeParserPtr &parser,
                                         const std::unordered_map<std::string, std::string> &exit_to_gather,
                                         const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                                        session::KernelWithIndexCmp> &graph_output_to_actor) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(parser);
  std::map<size_t, InputInfo> input_aids;
  if (actor->type() == KernelTransformType::kEntranceActor) {
    const auto &entrance_actor = dynamic_cast<EntranceActor *>(actor);
    MS_EXCEPTION_IF_NULL(entrance_actor);
    for (size_t i = 0; i < entrance_actor->formal_parameters().size(); ++i) {
      const auto &formal_parameter = entrance_actor->formal_parameters()[i];
      MS_EXCEPTION_IF_NULL(formal_parameter.first);
      input_aids[i] = {formal_parameter.first->DebugString(0), formal_parameter.second};
    }
    return input_aids;
  }
  size_t max_index = 0;
  // Get all inputs by input arrows.
  GetAllInputByArrow(actor, graph_output_to_actor, &input_aids, &max_index);
  // Get all inputs by device tensor store.
  for (const auto &pair : actor->device_tensor_store_keys()) {
    MS_EXCEPTION_IF_NULL(pair.second);
    std::string name = pair.second->DebugString(0);
    if (pair.second->isa<ValueNode>()) {
      name = GetValueNodeName(pair.second->cast<ValueNodePtr>());
    }
    input_aids[pair.first] = {name, 0};
    max_index = (max_index > pair.first ? max_index : pair.first);
  }

  // Get all inputs for control actor.
  if (IsControlFlowActor(actor->type())) {
    GetInputNameForControlActor(actor, &input_aids, &max_index);
  }
  for (auto &pair : input_aids) {
    const auto &iter = exit_to_gather.find(pair.second.name + actor->GetAID().Name());
    if (iter != exit_to_gather.end()) {
      MS_LOG(DEBUG) << "Replace input aid for dump from:" << pair.second.name
                    << " to:" << iter->second + kStubActorNameSuffix << " for actor:" << actor->GetAID();
      pair.second.name = iter->second + kStubActorNameSuffix;
    }
  }
  if (max_index == 0 || max_index + 1 == input_aids.size()) {
    return input_aids;
  }
  if (actor->type() == KernelTransformType::kUnknown && input_aids.size() <= max_index) {
    std::vector<size_t> invalid_index;
    for (size_t i = 0; i < max_index; ++i) {
      if (input_aids.find(i) == input_aids.end()) {
        invalid_index.emplace_back(i);
      }
    }
    if (input_aids.size() + invalid_index.size() == max_index + 1) {
      for (const auto &i : invalid_index) {
        input_aids[i] = {"para_U", 0};
      }
      return input_aids;
    }
  }
  for (const auto &pair : input_aids) {
    MS_LOG(DEBUG) << "index:" << pair.first << " input:" << pair.second.name << " index:" << pair.second.index;
  }
  MS_LOG(EXCEPTION) << " invalid input size:" << max_index + 1 << " for actor:" << actor->GetAID();
}

std::vector<AbstractActorPtr> InsertExecuteActor(std::vector<AbstractActor *> *actors,
                                                 const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(actors);
  MS_EXCEPTION_IF_NULL(parser);
  std::vector<AbstractActorPtr> new_actors;
  std::vector<AbstractActor *> old_actors = *actors;
  actors->clear();
  std::map<std::string, std::vector<KernelGraphPtr>> exit_actor_name_to_kernel_graphs;
  for (const auto &group : parser->kernel_graph_group_infos()) {
    MS_EXCEPTION_IF_NULL(group);
    std::vector<KernelGraphPtr> graphs(group->graphs_.begin(), group->graphs_.end());
    std::sort(graphs.begin(), graphs.end(), [](const KernelGraphPtr &graph1, const KernelGraphPtr &graph2) {
      MS_EXCEPTION_IF_NULL(graph1);
      MS_EXCEPTION_IF_NULL(graph2);
      return graph1->graph_id() < graph2->graph_id();
    });
    exit_actor_name_to_kernel_graphs[group->group_name_ + kExitActorNameSuffix] = graphs;
  }
  for (const auto &actor : old_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (!IsControlFlowActor(actor->type())) {
      MS_LOG(EXCEPTION) << "Invalid actor:" << actor->GetAID() << " for control actor dump.";
    }
    const auto &control_actor = dynamic_cast<ControlActor *>(actor);
    MS_EXCEPTION_IF_NULL(control_actor);
    if (control_actor->type() == KernelTransformType::kGatherActor && control_actor->node() != nullptr &&
        common::AnfAlgo::IsCallNode(control_actor->node())) {
      actors->emplace_back(actor);
      const auto &func_graphs = parser->FetchFuncGraphbyCallNode(control_actor->node());
      std::string actor_name = control_actor->GetAID().Name() + kStubActorNameSuffix;
      std::vector<FuncGraph *> graphs;
      std::for_each(func_graphs.begin(), func_graphs.end(),
                    [&graphs](const auto &func_graph) { graphs.emplace_back(func_graph.get()); });
      auto stub_actor = std::make_shared<StubActor>(actor_name, KernelTransformType::kUnknown, nullptr, graphs, "");
      MS_EXCEPTION_IF_NULL(stub_actor);
      auto data_arrow = std::make_shared<DataArrow>(0, stub_actor->GetAID(), 0);
      MS_EXCEPTION_IF_NULL(data_arrow);
      stub_actor->arrows_.emplace_back(data_arrow);
      stub_actor->AddInputAid(control_actor->GetAID(), data_arrow.get());
      new_actors.emplace_back(stub_actor);
      actors->emplace_back(stub_actor.get());
      continue;
    } else if (control_actor->node() == nullptr && control_actor->type() == KernelTransformType::kExitActor) {
      const auto &iter = exit_actor_name_to_kernel_graphs.find(control_actor->GetAID().Name());
      if (iter == exit_actor_name_to_kernel_graphs.end()) {
        MS_LOG(EXCEPTION) << "Invalid exit actor:" << control_actor->GetAID();
      }
      for (const auto &graph : iter->second) {
        MS_EXCEPTION_IF_NULL(graph);
        std::vector<FuncGraph *> graphs = {graph.get()};
        auto stub_actor =
          std::make_shared<StubActor>(control_actor->GetAID().Name() + graph->ToString() + kStubActorNameSuffix,
                                      KernelTransformType::kUnknown, nullptr, graphs, control_actor->GetAID().Name());
        MS_EXCEPTION_IF_NULL(stub_actor);
        new_actors.emplace_back(stub_actor);
        actors->emplace_back(new_actors.back().get());
        const auto &super_kernel_actor = FetchActor(graph->ToString() + kSuperKernelActorNameSuffix);
        if (super_kernel_actor != nullptr) {
          std::for_each(super_kernel_actor->input_data_arrow_aids().begin(),
                        super_kernel_actor->input_data_arrow_aids().end(),
                        [&stub_actor](const auto &pair) { stub_actor->AddInputAid(pair.first, pair.second); });
          stub_actor->set_device_tensor_store_keys(super_kernel_actor->device_tensor_store_keys());
        }
      }
    }
    actors->emplace_back(actor);
  }
  return new_actors;
}
}  // namespace

void ControlNodeScheduler::DumpControlActorInfo(
  const ExitActorPtr &exit_actor, const ControlNodeParserPtr &parser,
  const std::unordered_map<std::string, std::string> &exit_to_gather,
  const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>, session::KernelWithIndexCmp>
    &graph_output_to_actor,
  std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(parser);
  const auto &node = exit_actor->node();
  MS_EXCEPTION_IF_NULL(node);
  const auto &func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto get_input_aid_func = [this, &parser, &exit_to_gather, &func_graph](AbstractActor *const actor) {
    return GetInputAids(actor, parser, exit_to_gather, func_graph);
  };
  std::vector<AbstractActor *> actors = TopoSortForActor(exit_actor.get(), get_input_aid_func);
  auto new_actors = InsertExecuteActor(&actors, parser);
  MS_LOG(DEBUG) << "Topo sort size:" << actors.size() << " for exit actor:" << exit_actor->GetAID();
  ofs << "\nActor for func graph:" << func_graph->ToString() << "\n";
  size_t i = 0;
  std::unordered_map<std::string, size_t> relative_index;
  for (auto actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    ofs << "%" << std::to_string(i) << " = " << GetActorDumpName(actor) << "(";
    relative_index[actor->GetAID().Name()] = i++;
    const auto &input_names = GetInputName(actor, parser, exit_to_gather, graph_output_to_actor);
    for (const auto &pair : input_names) {
      const auto &input_name = pair.second.name;
      size_t input_index = pair.second.index;
      if (relative_index.find(input_name) == relative_index.end()) {
        ofs << input_name;
      } else {
        ofs << "%" << std::to_string(relative_index[input_name]) << "[" << std::to_string(input_index) << "]";
      }
      if (pair.first + 1 != input_names.size()) {
        ofs << ", ";
      }
    }
    ofs << ")\n";
  }
}

void CollectExitToGather(const std::vector<ExitActorPtr> &func_graph_exit_actors,
                         const std::map<int, std::string> &branch_id_to_aid,
                         std::unordered_map<std::string, std::string> *exit_actor_to_gather_actor) {
  MS_EXCEPTION_IF_NULL(exit_actor_to_gather_actor);
  for (const auto &exit_actor : func_graph_exit_actors) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    for (const auto &pair : exit_actor->output_branch_data_arrows()) {
      if (pair.first == 0) {
        continue;
      }
      if (branch_id_to_aid.find(pair.first) == branch_id_to_aid.end()) {
        MS_LOG(EXCEPTION) << "Invalid data arrow branch id:" << pair.first
                          << " for exit actor:" << exit_actor->GetAID();
      }
      for (const auto &arrow : pair.second) {
        MS_EXCEPTION_IF_NULL(arrow);
        (*exit_actor_to_gather_actor)[exit_actor->GetAID().Name() + arrow->to_op_id_.Name()] =
          branch_id_to_aid.at(pair.first);
      }
    }
    for (const auto &pair : exit_actor->output_branch_control_arrows()) {
      if (pair.first == 0) {
        continue;
      }
      if (branch_id_to_aid.find(pair.first) == branch_id_to_aid.end()) {
        MS_LOG(EXCEPTION) << "Invalid control arrow branch id:" << pair.first
                          << " for exit actor:" << exit_actor->GetAID();
      }
      for (const auto &aid : pair.second) {
        (*exit_actor_to_gather_actor)[exit_actor->GetAID().Name() + aid.Name()] = branch_id_to_aid.at(pair.first);
      }
    }

    for (const auto &pair : exit_actor->output_branch_partial_arrows()) {
      if (pair.first == 0) {
        continue;
      }
      if (branch_id_to_aid.find(pair.first) == branch_id_to_aid.end()) {
        MS_LOG(EXCEPTION) << "Invalid partial arrow branch id:" << pair.first
                          << " for exit actor:" << exit_actor->GetAID();
      }
      for (const auto &arrow : pair.second) {
        MS_EXCEPTION_IF_NULL(arrow);
        (*exit_actor_to_gather_actor)[exit_actor->GetAID().Name() + arrow->to_op_id_.Name()] =
          branch_id_to_aid.at(pair.first);
      }
    }
  }
}

void ControlNodeScheduler::DumpFormatControlActorSet(
  const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
  const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>, session::KernelWithIndexCmp>
    &graph_output_to_actor,
  std::ofstream &ofs) {
  if (actor_set == nullptr || actor_set->control_actors_ == nullptr ||
      actor_set->control_actors_->exit_actors_.empty() || graph_compiler_info.control_node_parser_ == nullptr ||
      !graph_compiler_info.control_node_parser_->IsInited()) {
    return;
  }
  try {
    MS_LOG(DEBUG) << "Dump format control actor set start.";
    std::vector<ExitActorPtr> func_graph_exit_actors;
    for (const auto &exit_actor : actor_set->control_actors_->exit_actors_) {
      if (exit_actor == nullptr || exit_actor->node() == nullptr) {
        continue;
      }
      func_graph_exit_actors.emplace_back(exit_actor);
    }
    std::map<int, std::string> branch_id_to_aid;
    for (const auto &pair : graph_compiler_info.control_node_parser_->call_node_to_branch_id_) {
      MS_EXCEPTION_IF_NULL(pair.first);
      branch_id_to_aid[pair.second] = GetActorName(pair.first);
    }
    std::unordered_map<std::string, std::string> exit_actor_to_gather_actor;
    CollectExitToGather(func_graph_exit_actors, branch_id_to_aid, &exit_actor_to_gather_actor);
    for (const auto &exit_actor : func_graph_exit_actors) {
      DumpControlActorInfo(exit_actor, graph_compiler_info.control_node_parser_, exit_actor_to_gather_actor,
                           graph_output_to_actor, ofs);
    }
    MS_LOG(DEBUG) << "Dump format control actor set end.";
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Dump format control actor failed, reason:" << e.what();
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
