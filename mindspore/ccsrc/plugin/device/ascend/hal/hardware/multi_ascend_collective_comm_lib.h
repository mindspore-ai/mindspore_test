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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include "runtime/collective/collective_communication_lib.h"
#include "runtime/collective/collective_comm_lib_loader.h"
#include "runtime/collective/communication_group.h"
#include "plugin/device/ascend/hal/hardware/multi_ascend_communication_group.h"
#include "plugin/device/ascend/hal/hardware/ascend_communication_group.h"
#ifdef ENABLE_INTERNAL_KERNELS
#include "plugin/device/ascend/hal/hardware/lowlatency_communication_group.h"
#endif
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "utils/dlopen_macro.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
constexpr char kMACCLGlobalGroupName[] = "hccl_world_group";

class EXPORT_WRAPPER MultiAscendCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MultiAscendCollectiveCommLib &GetInstance() {
    static MultiAscendCollectiveCommLib instance;
    return instance;
  }

  bool isGroupWithinLocalMachine(const std::vector<uint32_t> &group_ranks);

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size) override;

  bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool DestroyCommunicationGroup(const std::string &group_name) override;

  bool DestroyDeviceCommunicationGroup(const std::string &group_name) override;

  uint32_t GetRankId(const std::string &group_name) override;

  uint32_t GetGroupSize(const std::string &group_name) override;

  uint32_t GetLocalRankId(const std::string &group_name) override;

  uint32_t GetLocalGroupSize(const std::string &group_name) override;

  uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) override;

  uint32_t GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) override;

  std::unordered_set<std::string> GetLcclEnabledGroups();

 private:
  CollectiveCommunicationLib *ascend_collective_comm_lib_;
  CollectiveCommunicationLib *lowlatency_collective_comm_lib_;
  std::unordered_set<std::string> lccl_enabled_groups;
  MultiAscendCollectiveCommLib();
  ~MultiAscendCollectiveCommLib() override = default;
};
}  // namespace ascend

}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_
