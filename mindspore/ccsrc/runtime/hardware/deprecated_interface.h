/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_

#include <vector>
#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "utils/ms_context.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace device {
class DeprecatedInterface {
 public:
  virtual ~DeprecatedInterface() = default;

  // ascend
  virtual void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {}
  virtual bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) { return true; }
  virtual bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) { return true; }
  virtual bool IsTsdOpened(const std::shared_ptr<MsContext> &inst_context) { return true; }
  // gpu
  virtual int GetGPUCapabilityMajor() { return -1; }
  virtual int GetGPUCapabilityMinor() { return -1; }
  virtual int GetGPUMultiProcessorCount() { return -1; }
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_
