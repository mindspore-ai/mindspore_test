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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include "runtime/collective/communication_group.h"
#include "plugin/device/ascend/hal/hardware/ascend_communication_group.h"
#include "plugin/device/ascend/hal/hardware/lowlatency_communication_group.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace device {
namespace ascend {
class MultiAscendCommunicationGroup : public CommunicationGroup {
 public:
  explicit MultiAscendCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                         uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~MultiAscendCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  void SetHcclGroup(CommunicationGroupPtr hccl_group) { hccl_group_ = hccl_group; }
  CommunicationGroupPtr GetHcclGroup() { return hccl_group_; }
#ifdef ENABLE_INTERNAL_KERNELS
  void SetLcclGroup(CommunicationGroupPtr lccl_group) { lccl_group_ = lccl_group; }
  CommunicationGroupPtr GetLcclGroup() { return lccl_group_; }
#endif

 protected:
  //   MultiRootInfo root_info_;
  CommunicationGroupPtr hccl_group_;
  CommunicationGroupPtr lccl_group_;
};
using MultiAscendCommunicationGroupPtr = std::shared_ptr<MultiAscendCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COMMUNICATION_GROUP_H_
