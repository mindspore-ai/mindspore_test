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

// kernel_cache.h

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_EXECUTION_ORDER_CHECK_KERNEL_CACHE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_EXECUTION_ORDER_CHECK_KERNEL_CACHE_H_

#include <vector>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "ir/anf.h"

namespace mindspore {
namespace runtime {

class KernelCache {
 public:
  static KernelCache &GetInstance() {
    static KernelCache instance;
    return instance;
  }

  inline void Add(const CNodePtr &kernel) { current_buffer_.emplace_back(kernel); }

  void SwapBuffers(int step);

  std::vector<CNodePtr> GetBuffers(int step);

  bool need_add{false};

 private:
  KernelCache() = default;
  ~KernelCache() = default;
  KernelCache(const KernelCache &) = delete;
  KernelCache &operator=(const KernelCache &) = delete;

  std::vector<CNodePtr> current_buffer_;
  std::unordered_map<int, std::vector<CNodePtr>> step_buffers_;
  std::mutex mutex_;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_EXECUTION_ORDER_CHECK_KERNEL_CACHE_H_
