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

#include "plugin/device/ascend/hal/hardware/ascend_collective_comm/lowlatency_communication_group.h"

namespace mindspore {
namespace device {
namespace ascend {
std::vector<int> convertVectorUint32ToInt32(const std::vector<uint32_t> &global_ranks) {
  std::vector<int> outputs;
  for (auto &global_rank : global_ranks) {
    CHECK_RET((global_rank < INT_MAX), true, "The input global rank exceeds the limitation.");
    outputs.push_back(static_cast<int>(global_rank));
  }
  return outputs;
}

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

  uint32_t device_id = std::stoi(std::getenv("DEVICE_ID"));
  auto ret = aclrtSetDevice(device_id);
  if (ret != ACL_RT_SUCCESS) {
    return false;
  }
  uint32_t group_rank = GetGroupRank(global_rank_);

  // ADDING INPUT GROUP_RANKS
  std::vector<int> group_ranks_int = convertVectorUint32ToInt32(group_ranks_);
  lcal_comm_ = std::make_shared<LcalComm>(group_rank, size_, group_ranks_int);
  CHECK_IF_NULL(lcal_comm_);
  if (lcal_comm_->Init() != LCAL_SUCCESS) {
    return false;
  }

  lccl_comm_ = std::make_shared<Lccl>(*(lcal_comm_.get()));
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
