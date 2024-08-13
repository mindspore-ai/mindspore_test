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

#include "plugin/device/ascend/hal/hardware/multi_ascend_collective_comm_lib.h"

constexpr size_t kPathMax = 4096;
namespace mindspore {
namespace device {
namespace ascend {
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

MultiAscendCollectiveCommLib::MultiAscendCollectiveCommLib() { global_group_name_ = kMACCLGlobalGroupName; }

std::unordered_set<std::string> MultiAscendCollectiveCommLib::GetLcclEnabledGroups() { return lccl_enabled_groups; }

bool MultiAscendCollectiveCommLib::isGroupWithinLocalMachine(const std::vector<uint32_t> &group_ranks) {
  std::vector<size_t> all_host_hashs = distributed::collective::CollectiveManager::instance()->GetAllHostHashs();
  if (all_host_hashs.empty()) {
    MS_LOG(INFO) << "all_host_hashs is empty.";
    return false;
  }
  return std::all_of(group_ranks.begin() + 1, group_ranks.end(),
                     [&](uint32_t rank) { return all_host_hashs[rank] == all_host_hashs[0]; });
}

bool MultiAscendCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::EnableLccl()) {
    std::string lowlatency_comm_lib_name = GetCurrentDir() + "/ascend/liblowlatency_collective.so";
    auto loader = std::make_shared<CollectiveCommLibLoader>(lowlatency_comm_lib_name);
    MS_EXCEPTION_IF_NULL(loader);
    if (!loader->Initialize()) {
      MS_LOG(EXCEPTION) << "Loading LCCL collective library failed.";
      return false;
    }
    void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

    auto instance_func = DlsymFuncObj(lowlatency_communication_lib_instance, collective_comm_lib_handle);
    lowlatency_collective_comm_lib_ = instance_func();
    MS_EXCEPTION_IF_NULL(lowlatency_collective_comm_lib_);
    MS_LOG(WARNING) << "Loading LCCL because env MS_ENABLE_LCCL is set to on. Pay attention that LCCL only supports "
                       "single-node-multi-card mode in KernelByKernel for now.";
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->Initialize(global_rank, global_rank_size, local_rank_id),
                             "Failed to initialize LCCL.");
  }
#endif
  ascend_collective_comm_lib_ = &AscendCollectiveCommLib::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->Initialize(global_rank, global_rank_size, local_rank_id),
                           "Failed to initialize HCCL.");
  return true;
}

bool MultiAscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &group_name) {
  RETURN_IF_FALSE(ascend_collective_comm_lib_->DestroyDeviceCommunicationGroup(group_name));
  return true;
}

bool MultiAscendCollectiveCommLib::DestroyCommunicationGroup(const std::string &group_name) {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::EnableLccl() && lccl_enabled_groups.find(group_name) != lccl_enabled_groups.end()) {
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->DestroyCommunicationGroup(group_name),
                             "Failed to destroy LCCL communication group " + group_name);
    lccl_enabled_groups.erase(group_name);
  }
#endif
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->DestroyCommunicationGroup(group_name),
                           "Failed to destroy HCCL communication group " + group_name);
  return true;
}

bool MultiAscendCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                                  const std::vector<uint32_t> &group_ranks) {
  RETURN_IF_FALSE(ascend_collective_comm_lib_->CreateDeviceCommunicationGroup(group_name, group_ranks));
  return true;
}

bool MultiAscendCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                            const std::vector<uint32_t> &group_ranks,
                                                            uint32_t local_group_rank, uint32_t local_group_size) {
  MultiAscendCommunicationGroupPtr group = std::make_shared<MultiAscendCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size);
  MS_EXCEPTION_IF_NULL(group);
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::EnableLccl() && isGroupWithinLocalMachine(group_ranks)) {
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->CreateCommunicationGroup(
                               group_name, group_ranks, local_group_rank, local_group_size),
                             "Failed to create LCCL communication group" + group_name);
    CommunicationGroupPtr lccl_group = lowlatency_collective_comm_lib_->GetGroup(group_name);
    MS_EXCEPTION_IF_NULL(lccl_group);
    group->SetLcclGroup(lccl_group);
    lccl_enabled_groups.insert(group_name);
  }
#endif
  RETURN_IF_FALSE_WITH_LOG(
    ascend_collective_comm_lib_->CreateCommunicationGroup(group_name, group_ranks, local_group_rank, local_group_size),
    "Failed to create HCCL communication group" + group_name);
  CommunicationGroupPtr hccl_group = ascend_collective_comm_lib_->GetGroup(group_name);
  MS_EXCEPTION_IF_NULL(hccl_group);
  group->SetHcclGroup(hccl_group);

  groups_[group_name] = group;

  return true;
}

uint32_t MultiAscendCollectiveCommLib::GetRankId(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetRankId(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetGroupSize(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetGroupSize(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetLocalRankId(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetLocalRankId(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetLocalGroupSize(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetLocalGroupSize(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetWorldRankFromGroupRank(group_name, local_rank);
}

uint32_t MultiAscendCollectiveCommLib::GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetGroupRankFromWorldRank(world_rank, group_name);
}
}  // namespace ascend

using MultiAscendCollectiveCommLib = mindspore::device::ascend::MultiAscendCollectiveCommLib;

CollectiveCommunicationLib *multi_ascend_communication_lib_instance() {
  return &MultiAscendCollectiveCommLib::GetInstance();
}

}  // namespace device
}  // namespace mindspore
