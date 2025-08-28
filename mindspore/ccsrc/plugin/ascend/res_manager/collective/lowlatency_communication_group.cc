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

#include "plugin/ascend/res_manager/collective/lowlatency_communication_group.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {

LowlatencyCommunicationGroup::LowlatencyCommunicationGroup(const std::string &name,
                                                           const std::vector<uint32_t> &group_ranks,
                                                           uint32_t global_rank, uint32_t local_group_rank,
                                                           uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      lcal_comm_(nullptr),
      lccl_comm_(nullptr) {}

bool LowlatencyCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return true;
  }

  uint32_t device_id = std::stoi(common::EnvHelper::GetInstance()->GetEnv("DEVICE_ID"));
  auto ret = aclrtSetDevice(device_id);
  if (ret != ACL_RT_SUCCESS) {
    return false;
  }
  uint32_t group_rank = GetGroupRank(global_rank_);

  // ADDING INPUT GROUP_RANKS
  int group_rank_int = static_cast<int>(group_ranks_[0]);
  if (LcalCommInitRankWithDomain(group_rank_int, size_, group_rank, &lcal_comm_) != LCAL_SUCCESS) {
    return false;
  }
  CHECK_IF_NULL(lcal_comm_);
  lccl_comm_ = std::make_shared<Lccl>(*(static_cast<LcalComm *>(lcal_comm_)));
  CHECK_IF_NULL(lccl_comm_);

  initialized_ = true;
  return true;
}

bool LowlatencyCommunicationGroup::Finalize() {
  if (!initialized_) {
    return true;
  }
  initialized_ = false;
  return true;
}

void *LowlatencyCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(size_t);
  return root_info_size;
}

const LcclPtr &LowlatencyCommunicationGroup::lccl_communicator() const { return lccl_comm_; }

const LcalCommPtr &LowlatencyCommunicationGroup::lcal_comm() const { return lcal_comm_; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
