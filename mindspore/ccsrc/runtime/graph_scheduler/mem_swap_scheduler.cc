/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "runtime/graph_scheduler/mem_swap_scheduler.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <utility>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/offload_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/common/utils/comm_manager.h"
#include "runtime/device/gsm/swap_strategy_builder.h"
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/graph_scheduler/control_node_parser.h"
#include "runtime/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace runtime {
constexpr size_t kFirstVirtualNode = 0;
constexpr size_t kSecondVirtualNodeOffset = 1;
constexpr char kOffloadTargetCPU[] = "cpu";
constexpr char kOffloadTargetDisk[] = "disk";
namespace {
constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
constexpr char NCCL_WORLD_GROUP[] = "nccl_world_group";
size_t GetLocalRankSize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string world_group;
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    world_group = HCCL_WORLD_GROUP;
  } else if (backend == kGPUDevice) {
    world_group = NCCL_WORLD_GROUP;
  } else {
    MS_LOG(WARNING) << "Invalid communication backend: " << backend << ", currently only support Ascend/GPU backend.";
    return 1;
  }
  const auto &collective_manager = distributed::collective::CollectiveManager::instance();
  MS_EXCEPTION_IF_NULL(collective_manager);
  return collective_manager->GetLocalGroupSize(world_group);
}

ControlActor *GetCtrlActor(const ControlNodeParserPtr &parser, const KernelGraphPtr &graph,
                           const string &actor_suffix) {
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(graph);
  std::string actor_name;
  if (parser->IsInited() && !graph->execution_order().empty()) {
    actor_name = parser->FetchGroupNameByKernelGraph(graph) + actor_suffix;
  } else {
    actor_name = graph->ToString() + actor_suffix;
  }
  const auto kernel_graph_actor = reinterpret_cast<ControlActor *>(FetchActor(actor_name));
  if (kernel_graph_actor != nullptr) {
    return kernel_graph_actor;
  }
  const auto func_graph = parser->FetchFuncGraphByKernelGraph(graph.get());
  if (func_graph == nullptr) {
    return nullptr;
  }
  actor_name = func_graph->ToString() + actor_suffix;
  return reinterpret_cast<ControlActor *>(FetchActor(actor_name));
}

std::map<size_t, size_t> GetActionTensors(const std::shared_ptr<device::SwapAction> &swap_action,
                                          const std::shared_ptr<device::SwapStrategy> &swap_strategy,
                                          const HashMap<AnfNodePtr, size_t> &real_parameters,
                                          const KernelGraphPtr &graph,
                                          std::vector<DeviceTensor *> *fixed_device_address,
                                          std::vector<size_t> *real_parameter_index,
                                          std::vector<AnfNodePtr> *store_key) {
  MS_EXCEPTION_IF_NULL(swap_action);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  MS_EXCEPTION_IF_NULL(fixed_device_address);
  MS_EXCEPTION_IF_NULL(real_parameter_index);
  MS_EXCEPTION_IF_NULL(store_key);
  std::map<size_t, size_t> tensor_indexes_map;
  vector<vector<size_t>> tensor_ids_group(3, vector<size_t>());
  for (const auto &tensor_action : swap_action->actions_) {
    MS_EXCEPTION_IF_NULL(tensor_action);
    if (tensor_action->tensor_id_ >= swap_strategy->tensor_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid tensor id " << tensor_action->tensor_id_;
    }
    const auto &tensor_info = swap_strategy->tensor_infos_[tensor_action->tensor_id_];
    MS_EXCEPTION_IF_NULL(tensor_info);
    std::vector<size_t> real_tensor_ids;
    if (tensor_info->fused_tensor_ids_.empty()) {
      (void)real_tensor_ids.emplace_back(tensor_info->tensor_id_);
    } else {
      real_tensor_ids = tensor_info->fused_tensor_ids_;
    }
    for (const auto real_tensor_id : real_tensor_ids) {
      if (tensor_indexes_map.find(real_tensor_id) != tensor_indexes_map.end()) {
        continue;
      }
      if (real_tensor_id >= swap_strategy->tensor_infos_.size()) {
        MS_LOG(EXCEPTION) << "Invalid tensor id " << real_tensor_id;
      }
      const auto &real_tensor_info = swap_strategy->tensor_infos_[real_tensor_id];
      MS_EXCEPTION_IF_NULL(real_tensor_info);
      const auto &node = real_tensor_info->node_;
      const auto &real_parameter_iter = real_parameters.find(node);
      if (real_parameter_iter == real_parameters.end()) {
        if (node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
          auto front_node = GetFrontNodeByKernelGraph(node, graph.get());
          if (!front_node.first->isa<Parameter>() || front_node.second != 0) {
            MS_LOG(WARNING) << "Parameter " << node->DebugString()
                            << "'s front node is not a parameter: " << front_node.first->DebugString()
                            << ", index: " << front_node.second;
          } else {
            tensor_ids_group[2].emplace_back(real_tensor_id);
            store_key->emplace_back(front_node.first);
          }
        } else {
          const auto &output_addr = AnfAlgo::GetMutableOutputAddr(node, real_tensor_info->index_, false);
          tensor_ids_group[0].emplace_back(real_tensor_id);
          (void)fixed_device_address->emplace_back(output_addr.get());
        }
      } else {
        tensor_ids_group[1].emplace_back(real_tensor_id);
        (void)real_parameter_index->emplace_back(real_parameter_iter->second);
      }
    }
  }
  size_t input_idx = 0;
  for (const auto &tensor_ids : tensor_ids_group) {
    for (auto tensor_id : tensor_ids) {
      tensor_indexes_map[tensor_id] = input_idx++;
    }
  }
  return tensor_indexes_map;
}

void GenActionIndexList(const std::map<size_t, size_t> &tensors_id_index_map,
                        const std::shared_ptr<device::SwapAction> &swap_action,
                        const std::shared_ptr<device::SwapStrategy> &swap_strategy,
                        std::vector<std::pair<device::SwapActionType, vector<size_t>>> *actor_actions) {
  MS_EXCEPTION_IF_NULL(swap_action);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  MS_EXCEPTION_IF_NULL(actor_actions);
  std::vector<vector<size_t>> alloc_action_map;
  std::map<device::SwapActionType, vector<size_t>> move_action_map;
  const auto &actions = swap_action->actions_;
  for (const auto &tensor_action : actions) {
    MS_EXCEPTION_IF_NULL(tensor_action);
    if (tensor_action->tensor_id_ >= swap_strategy->tensor_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid tensor id " << tensor_action->tensor_id_;
    }
    const auto &tensor_info = swap_strategy->tensor_infos_[tensor_action->tensor_id_];
    MS_EXCEPTION_IF_NULL(tensor_info);
    if (tensor_action->action_ == device::SwapActionType::kAllocHBM) {
      std::vector<size_t> indexes;
      (void)std::transform(tensor_info->fused_tensor_ids_.begin(), tensor_info->fused_tensor_ids_.end(),
                           std::back_inserter(indexes),
                           [&](size_t tensor_id) { return tensors_id_index_map.at(tensor_id); });
      (void)alloc_action_map.emplace_back(indexes);
    } else if (tensor_action->action_ != device::SwapActionType::kUnDefined) {
      const auto tensor_id = tensor_info->tensor_id_;
      (void)move_action_map[tensor_action->action_].emplace_back(tensors_id_index_map.at(tensor_id));
    } else {
      MS_LOG(EXCEPTION) << "Undefined swap action type.";
    }
  }
  (void)std::transform(alloc_action_map.begin(), alloc_action_map.end(), std::back_inserter(*actor_actions),
                       [](const std::vector<size_t> &tensor_idxes) {
                         return std::make_pair(device::SwapActionType::kAllocHBM, tensor_idxes);
                       });
  (void)std::transform(move_action_map.begin(), move_action_map.end(), std::back_inserter(*actor_actions),
                       [](const std::pair<device::SwapActionType, vector<size_t>> &action) {
                         return std::make_pair(action.first, action.second);
                       });
}

std::shared_ptr<device::SwapContext> GetSwapContext() {
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto &offload_context = OffloadContext::GetInstance();
  MS_EXCEPTION_IF_NULL(offload_context);
  const auto &swap_context = std::make_shared<device::SwapContext>();
  const auto max_hbm_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
  swap_context->hbm_mem_size_ = FloatToSize(max_hbm_size * kGBToByte * offload_context->hbm_ratio());
  auto cpu_mem_size = offload_context->offload_cpu_size();
  if (!mindspore::IsStandAlone() && !offload_context->cpu_size_configured()) {
    size_t rank_size = GetLocalRankSize();
    cpu_mem_size = cpu_mem_size / rank_size;
  }
  swap_context->cpu_mem_size_ = static_cast<size_t>(cpu_mem_size * offload_context->cpu_ratio());
  swap_context->disk_mem_size_ = offload_context->offload_disk_size();
  MS_LOG(INFO) << "MOC size:" << swap_context->hbm_mem_size_ << ", cpu memory size:" << swap_context->cpu_mem_size_
               << ", disk size:" << swap_context->disk_mem_size_ << " to generate the offload strategy";
  if (!offload_context->auto_offload()) {
    const auto &offload_param = offload_context->offload_param();
    swap_context->offload_param_to_cpu_ = (offload_param == kOffloadTargetCPU);
    swap_context->offload_param_to_disk_ = (offload_param == kOffloadTargetDisk);
    const auto &offload_checkpoint = offload_context->offload_checkpoint();
    swap_context->offload_checkpoint_to_cpu_ = (offload_checkpoint == kOffloadTargetCPU);
    swap_context->offload_checkpoint_to_disk_ = (offload_checkpoint == kOffloadTargetDisk);
  }
  return swap_context;
}
}  // namespace

void MemSwapScheduler::GetRealParameters(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  ControlActor *source_actor = nullptr;
  if (parser->IsCallInputKernelGraph(graph.get())) {
    source_actor = GetCtrlActor(parser, graph, kStackActorNameSuffix);
  } else {
    source_actor = GetCtrlActor(parser, graph, kEntranceActorNameSuffix);
  }
  if (source_actor == nullptr) {
    return;
  }
  HashMap<AnfNodePtr, size_t> real_parameters;
  for (const auto &input : graph->input_nodes()) {
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (parser->IsControlFlowDataArrow(graph, input)) {
      const auto &front_node = GetFrontNodeByKernelGraph(input, graph.get());
      const size_t index = source_actor->FetchNodePosition(front_node);
      (void)real_parameters.insert({input, index});
    }
  }
  real_parameters_[graph->graph_id()].swap(real_parameters);
}

void MemSwapScheduler::BuildSwapActorForGraph(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                              const DeviceContext *device_context,
                                              std::vector<MemSwapActorPtr> *actors) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(actors);
  if (graph->is_dynamic_shape() || device_context->GetDeviceType() == device::DeviceType::kCPU) {
    return;
  }
  device::SwapStrategyBuilder builder;
  const auto &swap_context = GetSwapContext();
  auto swap_strategy = builder.Build(graph, swap_context);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  MS_LOG(INFO) << "Graph " << graph->graph_id() << ": " << swap_strategy->GetStatisticInfo();
  graph_strategy_map_[graph->graph_id()] = swap_strategy;

  if (swap_strategy->actions_.empty()) {
    return;
  }
  GetRealParameters(graph, parser);

  static size_t swap_actor_num = 0;
  const auto graph_id = graph->graph_id();
  for (const auto &iter : swap_strategy->actions_) {
    // Fixed DeviceAddress in MemorySwapActor.
    std::vector<DeviceTensor *> fixed_device_address;
    // Output index of EntranceActor or StackActor for real parameter whose DeviceAddress is not fixed.
    std::vector<size_t> real_parameter_index;
    std::vector<AnfNodePtr> store_key;
    // tensor_id - input_index
    auto tensors_id_index_map = GetActionTensors(iter.second, swap_strategy, real_parameters_[graph->graph_id()], graph,
                                                 &fixed_device_address, &real_parameter_index, &store_key);

    // SwapActionType - index of target DeviceAddress(fixed or changeable) in MemorySwapActor.
    std::vector<std::pair<device::SwapActionType, vector<size_t>>> actor_actions;
    GenActionIndexList(tensors_id_index_map, iter.second, swap_strategy, &actor_actions);

    const string swap_actor_name = kMemSwapActorNamePrefix + std::to_string(swap_actor_num++);
    auto swap_actor = std::make_shared<MemorySwapActor>(swap_actor_name, recorder_aid_, kDefaultStreamIndex,
                                                        fixed_device_address, device_context, actor_actions);

    size_t start_pos = real_parameter_index.size() + fixed_device_address.size();
    std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys;
    for (auto key : store_key) {
      MS_LOG(INFO) << "Using device tensor store key: " << key->DebugString() << " for " << swap_actor_name << ":"
                   << start_pos;
      device_tensor_store_keys.emplace_back(std::make_pair(start_pos++, key));
    }
    swap_actor->set_device_tensor_store_keys(device_tensor_store_keys);

    (void)actors->emplace_back(swap_actor);
    // Link data arrow from EntranceActor to MemorySwapActor later in Link
    data_dependency_[graph_id][swap_actor].swap(real_parameter_index);
    action_actor_map_[graph_id][iter.first] = swap_actor;
  }
}

std::vector<std::vector<MemSwapActorPtr>> MemSwapScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                  const AID *recorder_aid) {
  recorder_aid_ = recorder_aid;
  std::vector<std::vector<MemSwapActorPtr>> swap_actors;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return swap_actors;
  }
  const auto &offload_context = OffloadContext::GetInstance();
  MS_EXCEPTION_IF_NULL(offload_context);
  if (offload_context->specific_param_offload()) {
    MS_LOG(INFO) << "Offload strategy will not be generated when any parameter's device is set.";
    return swap_actors;
  }
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto device_context = graph_compiler_info.device_contexts_[i];
    const auto &graph = graph_compiler_info.graphs_[i];
    std::vector<MemSwapActorPtr> actors;
    if (device_context == nullptr || graph == nullptr || graph->is_dynamic_shape()) {
      swap_actors.emplace_back(actors);
      continue;
    }
    BuildSwapActorForGraph(graph, graph_compiler_info.control_node_parser_, device_context, &actors);
    swap_actors.emplace_back(actors);
  }
  return swap_actors;
}

AbstractActor *MemSwapScheduler::GetActorForLink(size_t id, const std::shared_ptr<device::SwapStrategy> &strategy,
                                                 const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                                 ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(strategy);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(actor_set);
  AbstractActor *ret = nullptr;
  if (id == kFirstVirtualNode) {
    ret = dynamic_cast<EntranceActor *>(GetCtrlActor(parser, graph, kEntranceActorNameSuffix));
    if (ret == nullptr) {
      ret = actor_set->data_prepare_actor_.get();
    }
  } else if (id == graph->execution_order().size() + kSecondVirtualNodeOffset) {
    ret = dynamic_cast<ExitActor *>(GetCtrlActor(parser, graph, kExitActorNameSuffix));
    if (ret == nullptr) {
      ret = actor_set->loop_count_actor_.get();
    }
  }
  if (ret != nullptr) {
    return ret;
  }
  const auto &node_iter = strategy->nodes_.find(id);
  if (node_iter != strategy->nodes_.end()) {
    const auto &node = node_iter->second;
    const auto kernel_type = FetchKernelTransformType(node, graph, {}, GraphExecutionStrategy::kPipeline);
    return FetchActor(kernel_type, actor_set->name_, node, graph);
  }
  const auto &action_actor_map = action_actor_map_.find(graph->graph_id());
  if (action_actor_map == action_actor_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find Actor for action id " << id << " in graph " << graph->graph_id();
  }
  const auto &action_iter = (action_actor_map->second).find(id);
  if (action_iter == (action_actor_map->second).end()) {
    MS_LOG(EXCEPTION) << "Can not find Actor for action id " << id << " in graph " << graph->graph_id();
  }
  return action_iter->second.get();
}

void MemSwapScheduler::LinkCtrlArrowForGraph(const std::shared_ptr<device::SwapStrategy> &strategy,
                                             const mindspore::KernelGraphPtr &graph,
                                             const mindspore::runtime::ControlNodeParserPtr &parser,
                                             mindspore::runtime::ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(strategy);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(actor_set);
  std::map<size_t, std::pair<size_t, size_t>> inplace_map;
  const size_t node_num = strategy->nodes_.size() - 1;
  for (const auto &iter : strategy->nodes_) {
    const auto id = iter.first;
    const auto &node = iter.second;
    if (IsSkippedKernelActor(node)) {
      inplace_map[id] = std::make_pair(id == 0 ? 1 : id - 1, id == node_num ? node_num - 1 : id + 1);
    }
  }
  for (const auto &link : strategy->links_) {
    MS_EXCEPTION_IF_NULL(link);
    size_t from_id = link->from_;
    const auto &from_iter = inplace_map.find(from_id);
    if (from_iter != inplace_map.end()) {
      from_id = from_iter->second.first;
    }
    const auto from_actor = GetActorForLink(from_id, strategy, graph, parser, actor_set);
    MS_EXCEPTION_IF_NULL(from_actor);
    size_t to_id = link->to_;
    const auto &to_iter = inplace_map.find(to_id);
    if (to_iter != inplace_map.end()) {
      to_id = to_iter->second.second;
    }
    const auto to_actor = GetActorForLink(to_id, strategy, graph, parser, actor_set);
    MS_EXCEPTION_IF_NULL(to_actor);
    if (from_actor != to_actor) {
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    }
  }
}

void MemSwapScheduler::LinkDataArrowForGraph(const std::shared_ptr<device::SwapStrategy> &strategy,
                                             const KernelGraphPtr &graph, const ControlNodeParserPtr &parser) const {
  ControlActor *source_actor = nullptr;
  if (parser->IsCallInputKernelGraph(graph.get())) {
    source_actor = GetCtrlActor(parser, graph, kStackActorNameSuffix);
  } else {
    source_actor = GetCtrlActor(parser, graph, kEntranceActorNameSuffix);
  }
  if (source_actor == nullptr) {
    return;
  }
  const auto &action_actor_map = action_actor_map_.find(graph->graph_id());
  if (action_actor_map == action_actor_map_.end()) {
    return;
  }
  const auto &data_dependency = data_dependency_.find(graph->graph_id());
  if (data_dependency == data_dependency_.end()) {
    return;
  }
  for (const auto &action_iter : strategy->actions_) {
    const auto &actor_iter = (action_actor_map->second).find(action_iter.first);
    if (actor_iter == (action_actor_map->second).end()) {
      continue;
    }
    const auto &data_dependency_iter = (data_dependency->second).find(actor_iter->second);
    if (data_dependency_iter == (data_dependency->second).end()) {
      continue;
    }
    for (size_t i = 0; i < data_dependency_iter->second.size(); ++i) {
      const auto &output_index = data_dependency_iter->second[i];
      SchedulerHelper::AddDataArrow(source_actor, actor_iter->second.get(), output_index, i);
    }
  }
}

void MemSwapScheduler::Link(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &parser = graph_compiler_info.control_node_parser_;
  for (const auto &graph : graph_compiler_info.graphs_) {
    const auto &strategy_iter = graph_strategy_map_.find(graph->graph_id());
    if (strategy_iter == graph_strategy_map_.end()) {
      continue;
    }
    const auto &strategy = strategy_iter->second;
    LinkCtrlArrowForGraph(strategy, graph, parser, actor_set);
    LinkDataArrowForGraph(strategy, graph, parser);
  }
}
}  // namespace runtime
}  // namespace mindspore
