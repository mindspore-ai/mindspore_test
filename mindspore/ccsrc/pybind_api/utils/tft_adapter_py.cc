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

#include <vector>
#include "include/common/pybind_api/api_register.h"
#include "include/backend/debug/tft_adapter/tft_wait_sem.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
using DeviceContext = mindspore::device::DeviceContext;
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
using DeviceTensorStore = mindspore::runtime::DeviceTensorStore;
using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
namespace {
DeviceContextPtr GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_name << " is not created yet.";
  }
  return device_ctx;
}
}  // namespace

bool GetMemUceInfo(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  return device_ctx->device_res_manager_->GetMemUceInfo(device_id);
}

std::vector<uint64_t> GetOptimizerTimestamps() {
  auto device_ctx = GetDeviceCtx();
  return device_ctx->device_res_manager_->GetOptimizerTimestamps();
}

bool GetUceLevelWithMemPoolForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                  const DeviceMemInfo &common_mem_blocks_info,
                                  const std::vector<std::pair<device::DeviceMemPtr, size_t>> &mem_uce_addr) {
  for (auto iter = persistent_mem_blocks_info.begin(); iter != persistent_mem_blocks_info.end(); ++iter) {
    void *persistent_block_start_addr = iter->first;
    auto block_info = iter->second.begin();
    void *persistent_block_end_addr = reinterpret_cast<char *>(persistent_block_start_addr) + block_info->second - 1;
    MS_EXCEPTION_IF_NULL(persistent_block_end_addr);
    for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
      void *mem_uce_start_addr = mem_uce_addr[i].first;
      void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
      MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
      if ((persistent_block_end_addr >= mem_uce_start_addr && persistent_block_start_addr <= mem_uce_start_addr) ||
          (mem_uce_end_addr >= persistent_block_start_addr && mem_uce_start_addr <= persistent_block_start_addr)) {
        MS_LOG(INFO) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }

  for (auto iter = common_mem_blocks_info.begin(); iter != common_mem_blocks_info.end(); ++iter) {
    void *common_block_start_addr = iter->first;
    auto block_info = iter->second.begin();
    void *common_block_end_addr = reinterpret_cast<char *>(common_block_start_addr) + block_info->second - 1;
    MS_EXCEPTION_IF_NULL(common_block_end_addr);
    for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
      void *mem_uce_start_addr = mem_uce_addr[i].first;
      void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
      MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
      if ((common_block_end_addr >= mem_uce_start_addr && common_block_start_addr <= mem_uce_start_addr) ||
          (mem_uce_end_addr >= common_block_start_addr && mem_uce_start_addr <= common_block_start_addr)) {
        MS_LOG(INFO) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }
  return false;
}

std::string GetUceProcessStrategyForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                        const DeviceMemInfo &common_mem_blocks_info,
                                        const std::vector<std::pair<device::DeviceMemPtr, size_t>> &mem_uce_addr) {
  // Judge whether weights got uce error.
  MS_LOG(INFO) << "Start to get UCE process strategy for kbk.";
  const auto &device_tensors = DeviceTensorStore::GetInstance().GetAll();
  try {
    for (auto iter = device_tensors.begin(); iter != device_tensors.end(); ++iter) {
      auto device_tensor_list = iter->second;
      for (const auto &device_tensor : device_tensor_list) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        void *device_tensor_start_addr = const_cast<void *>(device_tensor->GetPtr());
        void *device_tensor_end_addr =
          reinterpret_cast<char *>(device_tensor_start_addr) + device_tensor->GetSize() - 1;
        MS_EXCEPTION_IF_NULL(device_tensor_end_addr);
        for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
          void *mem_uce_start_addr = mem_uce_addr[i].first;
          void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
          MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
          // Return RS_UCE_HIGHLEVEL if overlap of device tensor addr and mem uce addr.
          if ((device_tensor_end_addr >= mem_uce_start_addr && device_tensor_start_addr <= mem_uce_start_addr) ||
              (mem_uce_end_addr >= device_tensor_start_addr && mem_uce_start_addr <= device_tensor_start_addr)) {
            MS_LOG(INFO) << "UCE process strategy is RS_UCE_HIGHLEVEL.";
            return device::RS_UCE_HIGHLEVEL;
          }
        }
      }
    }

    // Return RS_UCE_LOWLEVEL if overlap of memory pool addr and mem uce addr.
    if (GetUceLevelWithMemPoolForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_addr)) {
      return device::RS_UCE_LOWLEVEL;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "There is an error: " << e.what();
  }

  MS_LOG(INFO) << "UCE process strategy is RS_NORMAL.";

  return device::RS_NORMAL;
}

std::string GetUceProcessStrategy() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto persistent_mem_blocks_info = device_ctx->device_res_manager_->GetPersistentMemBlocksInfoStatistics();
  auto common_mem_blocks_info = device_ctx->device_res_manager_->GetCommonMemBlocksInfoStatistics();
  auto mem_uce_addr = device_ctx->device_res_manager_->GetMemUceAddr();
  return GetUceProcessStrategyForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_addr);
}

void UceMemRepair(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  device_ctx->device_res_manager_->UceMemRepair(device_id);
}

void StopDevice(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  MS_LOG(WARNING) << "Try to stop device: " << device_id;
  device_ctx->device_res_manager_->StopDevice(device_id);
  MS_LOG(WARNING) << "stop device: " << device_id << " end;";
}

void FinalizeCommunication() {
  MS_LOG(WARNING) << "Try to finalize communication";
  // Finalize HCCL_WORLD_GROUP
  auto ret =
    distributed::collective::CollectiveManager::instance()->DestroyDeviceSideCommunicationGroup(kHcclWorldGroup);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Destroy group:" << kHcclWorldGroup << " failed.";
  }
  // Finalize sub communication group
  MS_LOG(WARNING) << "Finalize sub communication";
  auto group_info = distributed::collective::CollectiveManager::instance()->get_group_info();
  for (const auto &item : group_info) {
    if (item.first == kHcclWorldGroup) {
      continue;
    }
    MS_LOG(WARNING) << "Destroy sub-group, group name: " << item.first << ", ranks: " << item.second;
    if (!distributed::collective::CollectiveManager::instance()->DestroyDeviceSideCommunicationGroup(item.first)) {
      MS_LOG(EXCEPTION) << "Destroy group:" << item.first << " failed, ranks: " << item.second;
    }
    MS_LOG(WARNING) << "Destroy sub-group, group name: " << item.first << " ok";
  }
  MS_LOG(WARNING) << "Finalize communication end";
}

void RebuildHcclWorldGroup() {
  MS_LOG(WARNING) << "Try to rebuild hccl world group communication";
  auto ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(kHcclWorldGroup);
  auto ret = distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(kHcclWorldGroup, ranks);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Rebuild group:" << kHcclWorldGroup << " failed, ranks: " << ranks;
  }
  MS_LOG(WARNING) << "Try to rebuild hccl world group communication ok";
}

void RebuildSubCommunication() {
  // rebuild sub comm
  MS_LOG(WARNING) << "Try to rebuild sub communication";
  auto group_info = distributed::collective::CollectiveManager::instance()->get_group_info();
  for (const auto &item : group_info) {
    if (item.first == kHcclWorldGroup) {
      continue;
    }
    MS_LOG(WARNING) << "Rebuild sub-group, group name: " << item.first << ", ranks: " << item.second;
    if (!distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(item.first, item.second)) {
      MS_LOG(EXCEPTION) << "Rebuild group:" << item.first << " failed, ranks: " << item.second;
    }
    MS_LOG(WARNING) << "Rebuild sub-group, group name: " << item.first << " ok";
  }
  UCEException::GetInstance().set_force_stop_flag(false);
  UCEException::GetInstance().clear_uce_error();
  MS_LOG(WARNING) << "Rebuild communication end";
}

void RegTFT(py::module *m) {
  (void)m->def("_stop_device", &mindspore::StopDevice, "Stop the device.");
  (void)m->def("_repair_device", &mindspore::UceMemRepair, "Repair the device.");
  (void)m->def("_get_uce_process_strategy", &mindspore::GetUceProcessStrategy, "Get UCE process strategy.");
  (void)m->def("_get_uce_mem_info", &mindspore::GetMemUceInfo, "Get UCE mem info.");
  (void)m->def("_get_optimzer_timestamps", &mindspore::GetOptimizerTimestamps,
               "Get optimizer start and finish timestamps.");
  (void)m->def(
    "_tft_sem_post", []() { mindspore::debug::tft::TFTWaitSem::GetInstance().Post(); }, "TFT sem start post");
  (void)m->def(
    "_tft_sem_enable", []() { mindspore::debug::tft::TFTWaitSem::Enable(); }, "TFT enable sem feature");
  (void)m->def(
    "_tft_start_record_threads", []() { mindspore::debug::tft::TFTWaitSem::GetInstance().StartRecordThreads(); },
    "TFT start recording newly created threads");
  (void)m->def(
    "_tft_finish_record_threads", []() { mindspore::debug::tft::TFTWaitSem::GetInstance().FinishRecordThreads(); },
    "TFT finish recording newly created threads");
  (void)m->def("_finalize_comm", &FinalizeCommunication, "Finalize comm.");
  (void)m->def("_rebuild_sub_group", &RebuildSubCommunication, "Rebuild comm.");
  (void)m->def("_rebuild_world_group", &RebuildHcclWorldGroup, "Rebuild comm.");
}
}  // namespace mindspore
