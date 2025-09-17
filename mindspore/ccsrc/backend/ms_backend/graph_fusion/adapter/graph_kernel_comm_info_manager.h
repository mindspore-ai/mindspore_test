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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_COMM_INFO_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_COMM_INFO_MANAGER_H_
#include <map>
#include <functional>
#include <memory>
#include <utility>
#include <string>

#include "base/base.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace graphkernel {
class BACKEND_EXPORT GraphKernelCommInfo {
 public:
  GraphKernelCommInfo() = default;
  virtual ~GraphKernelCommInfo() = default;
  virtual bool EnableComm() { return false; }
  virtual bool IsTargetCommOp(const AnfNodePtr op) { return false; }
};

using GraphKernelCommInfoCreator = std::function<std::shared_ptr<GraphKernelCommInfo>()>;

class BACKEND_EXPORT GraphKernelCommInfoManager {
 public:
  static GraphKernelCommInfoManager &Instance();
  void Register(const std::string &device_type, GraphKernelCommInfoCreator &&creator);
  void Clear() { comm_info_map_.clear(); }
  std::shared_ptr<GraphKernelCommInfo> GetCommInfo(const std::string &device_type);

 private:
  GraphKernelCommInfoManager() = default;
  virtual ~GraphKernelCommInfoManager() = default;
  std::map<std::string, GraphKernelCommInfoCreator> comm_info_map_;
};

class BACKEND_EXPORT GraphKernelCommInfoRegister {
 public:
  GraphKernelCommInfoRegister(const std::string &device_type, GraphKernelCommInfoCreator &&creator) {
    GraphKernelCommInfoManager::Instance().Register(device_type, std::move(creator));
  }
  ~GraphKernelCommInfoRegister() = default;
};

#define REG_GRAPH_KERNEL_COMM_INFO(DEVICE_TYPE, BUILDER_CLASS)                           \
  static const GraphKernelCommInfoRegister g_graph_kernel_comm_info_##DEVICE_TYPE##_reg( \
    DEVICE_TYPE, []() { return std::make_shared<BUILDER_CLASS>(); })
}  // namespace graphkernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_COMM_INFO_MANAGER_H_
