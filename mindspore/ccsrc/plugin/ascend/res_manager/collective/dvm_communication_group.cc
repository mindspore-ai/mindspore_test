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

#include "plugin/ascend/res_manager/collective/dvm_communication_group.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mindspore {
namespace device {
namespace ascend {

DvmCommunicationGroup::DvmCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                             uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size), dvm_comm_(nullptr) {}

bool DvmCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return true;
  }

  uint32_t device_id = std::stoi(std::getenv("DEVICE_ID"));
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  uint32_t group_rank = GetGroupRank(global_rank_);
  dvm_comm_ = std::make_shared<dvm::Comm>();
  CHECK_IF_NULL(dvm_comm_);
  const size_t kMinimumGroupSize = 2;
  if (local_group_size_ < kMinimumGroupSize) {
    return true;
  }
  if (!dvm_comm_->Init(group_rank, size_)) {
    MS_LOG(ERROR) << "init failed";
    return false;
  }
  initialized_ = true;
  MS_LOG(WARNING) << "Init success";
  return true;
}

bool DvmCommunicationGroup::Finalize() {
  if (!initialized_) {
    return true;
  }
  initialized_ = false;
  return true;
}

void *DvmCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(size_t);
  return root_info_size;
}

const CommPtr &DvmCommunicationGroup::dvm_communicator() const { return dvm_comm_; }

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
