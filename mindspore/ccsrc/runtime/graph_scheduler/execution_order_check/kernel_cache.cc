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
// kernel_cache.cpp
#include <utility>
#include "runtime/graph_scheduler/execution_order_check/kernel_cache.h"

namespace mindspore {
namespace runtime {
void KernelCache::SwapBuffers(int step) {
  std::lock_guard<std::mutex> lock(mutex_);
  step_buffers_[step] = std::move(current_buffer_);
  current_buffer_.clear();
  current_buffer_.reserve(100000);
}

std::vector<CNodePtr> KernelCache::GetBuffers(int step) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = step_buffers_.find(step);
  if (it != step_buffers_.end()) {
    return it->second;
  }
  return {};
}
}  // namespace runtime
}  // namespace mindspore
