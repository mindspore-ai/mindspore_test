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

#include "runtime/device/pre_launch_comm.h"

#include <algorithm>
#include <unordered_set>
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/device_address.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "mindspore/ops/op_def/framework_ops.h"

namespace mindspore {
namespace runtime {
PreLaunchComm &PreLaunchComm::GetInstance() {
  static PreLaunchComm instance;
  return instance;
}

CommKernelInfo PreLaunchComm::GetKernelInfo(const CNodePtr &kernel_node) {
  CommKernelInfo hccl_kernel_info;
  hccl_kernel_info.name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (common::AnfAlgo::HasNodeAttr(kAttrGroup, kernel_node)) {
    hccl_kernel_info.group = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrGroup);
  }
  std::vector<uint32_t> group_rank_ids =
    common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, kernel_node)
      ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(kernel_node, kAttrGroupRankIds)
      : std::vector<uint32_t>();
  if (group_rank_ids.empty()) {
    MS_LOG(DEBUG) << "The group_rank_ids of kernel: " << kernel_node->fullname_with_scope() << " is empty.";
    std::string group = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrGroup);
    group_rank_ids = distributed::collective::CollectiveManager::instance()->GetGroupRanks(group);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrSrcRank, kernel_node)) {
    auto src_rank_id = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrSrcRank);
    if (src_rank_id < 0 || static_cast<size_t>(src_rank_id) >= group_rank_ids.size()) {
      MS_LOG(EXCEPTION) << "The src_rank_id " << src_rank_id << " is out of range.";
    }
    hccl_kernel_info.src_rank = group_rank_ids[static_cast<size_t>(src_rank_id)];
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrDestRank, kernel_node)) {
    auto dest_rank_id = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrDestRank);
    if (dest_rank_id < 0 || static_cast<size_t>(dest_rank_id) >= group_rank_ids.size()) {
      MS_LOG(EXCEPTION) << "The dest_rank_id " << dest_rank_id << " is out of range.";
    }
    hccl_kernel_info.dest_rank = group_rank_ids[static_cast<size_t>(dest_rank_id)];
  }
  return hccl_kernel_info;
}

void PreLaunchComm::Launch(std::vector<LaunchCommNode> &launch_hccl_nodes, SortedFunc sorted) {
  auto sorted_func = [&sorted](const auto &a, const auto &b) {
    const auto &kernel_info_a = std::get<kIndex2>(a);
    const auto &kernel_info_b = std::get<kIndex2>(b);

    if (sorted == SORTED_BY_SEND_SEQUENTAIL) {
      return kernel_info_a.dest_rank < kernel_info_b.dest_rank;
    } else if (sorted == SORTED_BY_SEND_REVERSE) {
      return kernel_info_a.dest_rank > kernel_info_b.dest_rank;
    } else if (sorted == SORTED_BY_RECV_SEQUENTAIL) {
      return kernel_info_a.src_rank < kernel_info_b.src_rank;
    } else {
      return kernel_info_a.src_rank > kernel_info_b.src_rank;
    }
  };
  std::sort(launch_hccl_nodes.begin(), launch_hccl_nodes.end(), sorted_func);
  for (auto &launch_hccl_node : launch_hccl_nodes) {
    auto &kernel_actor = std::get<kIndex0>(launch_hccl_node);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto &kernel = std::get<kIndex1>(launch_hccl_node);
    auto &hccl_kernel_launch_info = std::get<kIndex3>(launch_hccl_node);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
    std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);
    std::vector<device::DeviceAddressPtr> input_device_addresses;
    std::vector<device::DeviceAddressPtr> output_device_addresses;
    auto device_context = kernel_actor->device_contexts()[kIndex0];
    MS_EXCEPTION_IF_NULL(device_context);
    auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "AllocMemoryForPreLaunchComm", "ConstructInputAndOutput",
                                                   "", false);
    for (const auto &input_kernel_tensor : input_kernel_tensors) {
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      auto kernel_tensor = std::make_shared<KernelTensor>(*input_kernel_tensor);
      const auto &new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "AllocMemoryForPreLaunchComm",
                                                     device::tracker::MemType::kOther, new_device_tensor->GetSize(),
                                                     new_device_tensor.get());
      device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), stream_id);
      input_kernel_tensor->set_device_ptr(new_device_tensor->GetMutablePtr());
      (void)input_device_addresses.emplace_back(new_device_tensor);
    }
    for (const auto &output_kernel_tensor : output_kernel_tensors) {
      MS_EXCEPTION_IF_NULL(output_kernel_tensor);
      auto kernel_tensor = std::make_shared<KernelTensor>(*output_kernel_tensor);
      const auto &new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "AllocMemoryForPreLaunchComm",
                                                     device::tracker::MemType::kOther, new_device_tensor->GetSize(),
                                                     new_device_tensor.get());
      device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), stream_id);
      output_kernel_tensor->set_device_ptr(new_device_tensor->GetMutablePtr());
      (void)output_device_addresses.emplace_back(new_device_tensor);
    }
    MS_LOG(DEBUG) << "Pre build hccl kernel " << kernel->fullname_with_scope();
    (void)kernel_mod->Launch(input_kernel_tensors, hccl_kernel_launch_info.workspace_kernel_tensors_,
                             output_kernel_tensors, hccl_kernel_launch_info.stream_);
    for (size_t i = 0; i < input_kernel_tensors.size(); i++) {
      auto kernel_tensor = input_kernel_tensors[i];
      auto device_address = input_device_addresses[i];
      device_context->device_res_manager_->FreeMemory(device_address.get());
      kernel_tensor->set_device_ptr(nullptr);
    }
    for (size_t i = 0; i < output_kernel_tensors.size(); i++) {
      auto kernel_tensor = output_kernel_tensors[i];
      auto device_address = output_device_addresses[i];
      device_context->device_res_manager_->FreeMemory(device_address.get());
      kernel_tensor->set_device_ptr(nullptr);
    }
  }
}

void PreLaunchComm::SpiltBucket(const std::vector<LaunchCommNode> &launch_hccl_nodes,
                                std::vector<LaunchCommNode> *hccl_nodes_vec_send_after_bucket,
                                std::vector<LaunchCommNode> *hccl_nodes_vec_receive_before_bucket,
                                std::vector<LaunchCommNode> *hccl_nodes_vec_send_before_bucket,
                                std::vector<LaunchCommNode> *hccl_nodes_vec_receive_after_bucket) {
  MS_EXCEPTION_IF_NULL(hccl_nodes_vec_send_after_bucket);
  MS_EXCEPTION_IF_NULL(hccl_nodes_vec_receive_before_bucket);
  MS_EXCEPTION_IF_NULL(hccl_nodes_vec_send_before_bucket);
  MS_EXCEPTION_IF_NULL(hccl_nodes_vec_receive_after_bucket);
  static auto rank = common::GetEnv(kRankID);
  if (rank.empty()) {
    return;
  }
  int64_t rank_id = std::stoi(rank);
  for (auto &launch_hccl_node : launch_hccl_nodes) {
    const auto &kernel_info = std::get<kIndex2>(launch_hccl_node);
    if (kernel_info.name == prim::kPrimSend->name() && kernel_info.dest_rank > rank_id) {
      hccl_nodes_vec_send_after_bucket->push_back(launch_hccl_node);
    } else if (kernel_info.name == prim::kPrimReceive->name() && kernel_info.src_rank < rank_id) {
      hccl_nodes_vec_receive_before_bucket->push_back(launch_hccl_node);
    } else if (kernel_info.name == prim::kPrimSend->name() && kernel_info.dest_rank < rank_id) {
      hccl_nodes_vec_send_before_bucket->push_back(launch_hccl_node);
    } else {
      hccl_nodes_vec_receive_after_bucket->push_back(launch_hccl_node);
    }
  }
}

void PreLaunchComm::PreLaunchCommKernel(runtime::ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (UseSimulationApi()) {
    return;
  }
  const PrimitiveSet prim_set{prim::kPrimSend, prim::kPrimReceive};
  std::vector<LaunchCommNode> pre_build_hccl_kernels;
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    if (kernel_actor == nullptr) {
      continue;
    }
    auto kernel = kernel_actor->kernel();
    if (IsOneOfPrimitiveCNode(kernel, prim_set)) {
      const auto &kernel_info = GetKernelInfo(kernel);
      pre_build_hccl_kernels.push_back(
        std::make_tuple(kernel_actor, kernel, kernel_info, kernel_actor->kernel_launch_info()));
    }
  }
  for (const auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    if (super_kernel_actor == nullptr) {
      continue;
    }
    for (const auto &kernel_actor : super_kernel_actor->kernel_actors()) {
      if (kernel_actor == nullptr) {
        continue;
      }
      auto kernel = kernel_actor->kernel();
      if (IsOneOfPrimitiveCNode(kernel, prim_set)) {
        const auto &kernel_info = GetKernelInfo(kernel);
        pre_build_hccl_kernels.push_back(
          std::make_tuple(kernel_actor, kernel, kernel_info, kernel_actor->kernel_launch_info()));
      }
    }
  }
  if (pre_build_hccl_kernels.empty()) {
    MS_LOG(INFO) << "No hccl kernel to pre launch";
    return;
  }
  // Remove duplicates
  struct TupleHash {
    std::size_t operator()(const LaunchCommNode &t) const {
      const auto &kernel_info = std::get<kIndex2>(t);
      return std::hash<std::string>()(kernel_info.ToString());
    }
  };

  struct TupleEqual {
    bool operator()(const LaunchCommNode &t1, const LaunchCommNode &t2) const {
      const auto &kernel_info_a = std::get<kIndex2>(t1);
      const auto &kernel_info_b = std::get<kIndex2>(t2);
      return kernel_info_a == kernel_info_b;
    }
  };
  std::unordered_set<LaunchCommNode, TupleHash, TupleEqual> launch_hccl_nodes(pre_build_hccl_kernels.begin(),
                                                                              pre_build_hccl_kernels.end());
  // Launch in specific order
  std::vector<LaunchCommNode> launch_hccl_nodes_vec(launch_hccl_nodes.begin(), launch_hccl_nodes.end());
  std::vector<LaunchCommNode> hccl_nodes_vec_send_after_bucket;
  std::vector<LaunchCommNode> hccl_nodes_vec_receive_before_bucket;
  std::vector<LaunchCommNode> hccl_nodes_vec_send_before_bucket;
  std::vector<LaunchCommNode> hccl_nodes_vec_receive_after_bucket;
  SpiltBucket(launch_hccl_nodes_vec, &hccl_nodes_vec_send_after_bucket, &hccl_nodes_vec_receive_before_bucket,
              &hccl_nodes_vec_send_before_bucket, &hccl_nodes_vec_receive_after_bucket);
  Launch(hccl_nodes_vec_send_after_bucket, SORTED_BY_SEND_SEQUENTAIL);
  Launch(hccl_nodes_vec_receive_before_bucket, SORTED_BY_RECV_REVERSE);
  Launch(hccl_nodes_vec_send_before_bucket, SORTED_BY_SEND_REVERSE);
  Launch(hccl_nodes_vec_receive_after_bucket, SORTED_BY_RECV_SEQUENTAIL);
}
}  // namespace runtime
}  // namespace mindspore
