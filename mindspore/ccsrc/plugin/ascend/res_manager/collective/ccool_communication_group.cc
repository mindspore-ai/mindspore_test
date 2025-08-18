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

#include "plugin/ascend/res_manager/collective/ccool_communication_group.h"

constexpr uint16_t kBeginBasePort = 21234;
constexpr uint16_t kConnMaxPort = 65535;
constexpr uint16_t kConnPortStride = 10;

namespace mindspore {
namespace device {
namespace ascend {
CcoolCommunicationGroup::CcoolCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                                 uint32_t global_rank, uint32_t local_group_rank,
                                                 uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size) {
  // local_group_rank is useless in CCOOL
  host_group_name_ = name + "_cross_az";
}

// The whole Initialize function is called by ExecuteFuncInThread in thread
bool CcoolCommunicationGroup::Initialize(void *root_info) {
  MS_LOG(WARNING) << "Start to init ccool communication group";
  MS_EXCEPTION_IF_NULL(hccl_root_info_);

  return hccl_group_->Initialize(hccl_root_info_);
}

bool CcoolCommunicationGroup::Finalize() { return true; }

void *CcoolCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  // generate hccl root info and save
  MS_EXCEPTION_IF_NULL(hccl_group_);
  hccl_root_info_ = hccl_group_->GenerateRootInfo(&hccl_root_info_size_);
  bool ret = false;
  while (!ret) {
    if (!host_comm_lib_instance_->BroadcastUniqueID(host_group_name_, hccl_root_info_size_, hccl_root_info_)) {
      MS_LOG(WARNING) << "ccool host comm lib BroadcastUniqueID fail, retry.";
    }
    ret = true;
  }
  // do nothing
  static char fake_root_info[] = "CcoolFakeRootInfo";
  *root_info_size = sizeof(fake_root_info);
  MS_LOG(WARNING) << "ccool generate root info, root info = " << fake_root_info << ", size = " << *root_info_size;
  return static_cast<void *>(fake_root_info);
}

bool CcoolCommunicationGroup::InitAscendCommGroup(const std::vector<std::string> &rank_az_map,
                                                  const std::vector<std::string> &rank_ip_map) {
  // init inner_cluster_ranks_ and inter_cluster_ranks_
  MS_LOG(WARNING) << "init ascend comm group, global_rank_ = " << global_rank_ << ", rank_az_map = " << rank_az_map;
  rank_az_map_ = rank_az_map;
  rank_ip_map_ = rank_ip_map;
  az_id_ = rank_az_map_[global_rank_];
  for (uint32_t rank : group_ranks_) {
    std::string az_id = rank_az_map_[rank];
    if (group_az_rank_map_.count(az_id) == 0) {
      group_az_rank_map_[az_id] = std::vector<uint32_t>();
    }
    group_az_rank_map_[az_id].push_back(rank);
  }
  inner_cluster_ranks_ = group_az_rank_map_[az_id_];

  auto iter = std::find(inner_cluster_ranks_.begin(), inner_cluster_ranks_.end(), global_rank_);
  uint32_t hccl_rank_id_ = static_cast<uint32_t>(std::distance(inner_cluster_ranks_.begin(), iter));
  // only support symmetric now
  for (const auto &group_az_ranks : group_az_rank_map_) {
    inter_cluster_ranks_.push_back(group_az_ranks.second[hccl_rank_id_]);
  }
  std::sort(inter_cluster_ranks_.begin(), inter_cluster_ranks_.end());

  MS_LOG(INFO) << "init ascend comm group, inner cluster ranks = " << inner_cluster_ranks_
               << ", inter cluster ranks = " << inter_cluster_ranks_ << ", hccl_rank_id = " << hccl_rank_id_;

  host_group_name_ += az_id_;
  RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->CreateCommunicationGroup(
                             host_group_name_, inner_cluster_ranks_, hccl_rank_id_, inner_cluster_ranks_.size()),
                           "Failed to create host communication group" + host_group_name_);

  RETURN_IF_FALSE_WITH_LOG(AscendCollectiveCommLib::GetInstance().CreateCommunicationGroup(
                             name_, inner_cluster_ranks_, hccl_rank_id_, inner_cluster_ranks_.size()),
                           "Failed to create device communication group" + name_);
  hccl_group_ = AscendCollectiveCommLib::GetInstance().GetGroup(name_);

  return true;
}

const std::vector<uint32_t> &CcoolCommunicationGroup::GetInterClusterRanks() const { return inter_cluster_ranks_; }

const std::vector<uint32_t> &CcoolCommunicationGroup::GetInnerClusterRanks() const { return inner_cluster_ranks_; }

LeaperConnInfo &CcoolCommunicationGroup::GetConnInfo(uint32_t dst_rank) {
  std::hash<std::string> hash_fn;
  uint16_t base_port = (hash_fn(this->name_) % kConnMaxPort) + kBeginBasePort;
  uint16_t src_port = base_port + kConnPortStride * global_rank_ + dst_rank;
  uint16_t dst_port = base_port + kConnPortStride * dst_rank + global_rank_;
  MS_LOG(INFO) << "src_port = " << src_port << ", dst_port = " << dst_port;
  if (rank_conn_info_map_.count(dst_rank) == 0) {
    rank_conn_info_map_[dst_rank] =
      LeaperTrans::GetInstance().Connect(rank_ip_map_[global_rank_], rank_ip_map_[dst_rank], src_port, dst_port);
  }
  return rank_conn_info_map_[dst_rank];
}

void CcoolCommunicationGroup::SetHostCommLib(CollectiveCommunicationLib *comm_lib) {
  MS_EXCEPTION_IF_NULL(comm_lib);
  host_comm_lib_instance_ = comm_lib;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
