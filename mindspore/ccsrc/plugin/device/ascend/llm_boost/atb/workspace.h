/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_WORKSPACE_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_WORKSPACE_H
#include <cstdint>
#include <memory>
#include <vector>
#include "plugin/device/ascend/llm_boost/atb/buffer_device.h"

namespace mindspore {
namespace kernel {
class Workspace {
 public:
  Workspace();
  ~Workspace();
  void *GetWorkspaceBuffer(uint64_t bufferSize);

 private:
  uint64_t GetWorkspaceBufferRing() const;
  uint64_t GetWorkspaceBufferSize() const;

 private:
  std::vector<std::unique_ptr<BufferDevice>> workspaceBuffers_;
  size_t workspaceBufferOffset_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_WORKSPACE_H
