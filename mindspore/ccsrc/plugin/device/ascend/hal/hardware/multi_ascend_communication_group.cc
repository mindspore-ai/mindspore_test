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

#include "plugin/device/ascend/hal/hardware/multi_ascend_communication_group.h"

namespace mindspore {
namespace device {
namespace ascend {
MultiAscendCommunicationGroup::MultiAscendCommunicationGroup(const std::string &name,
                                                             const std::vector<uint32_t> &group_ranks,
                                                             uint32_t global_rank, uint32_t local_group_rank,
                                                             uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      hccl_group_(nullptr),
      lccl_group_(nullptr) {}

bool MultiAscendCommunicationGroup::Initialize(void *root_info) {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::EnableLccl() && lccl_group_ != nullptr) {
    if (!lccl_group_->Initialize(root_info)) {
      MS_LOG(ERROR) << "Failed to initialize LCCL group " << name_;
      return false;
    }
    MS_LOG(INFO) << "Successfully initialize LCCL group " << name_;
  }
#endif
  if (!hccl_group_->Initialize(root_info)) {
    MS_LOG(ERROR) << "Failed to initialize HCCL group " << name_;
    return false;
  }
  MS_LOG(INFO) << "Successfully initialize HCCL group " << name_;
  return true;
}

bool MultiAscendCommunicationGroup::Finalize() {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::EnableLccl() && lccl_group_ != nullptr) {
    if (!lccl_group_->Finalize()) {
      MS_LOG(ERROR) << "Failed to finalize LCCL group" << name_;
      return false;
    }
    MS_LOG(INFO) << "Successfully finalize LCCL group " << name_;
  }
#endif
  if (!hccl_group_->Finalize()) {
    MS_LOG(ERROR) << "Failed to finalize HCCL group" << name_;
    return false;
  }
  MS_LOG(INFO) << "Successfully finalize HCCL group " << name_;
  return true;
}

void *MultiAscendCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  MS_EXCEPTION_IF_NULL(hccl_group_);
  void *root_info = hccl_group_->GenerateRootInfo(root_info_size);
  MS_EXCEPTION_IF_NULL(root_info);
  return root_info;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
