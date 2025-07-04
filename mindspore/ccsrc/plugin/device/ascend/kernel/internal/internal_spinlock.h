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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_INTERNAL_SPINLOCK_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_INTERNAL_SPINLOCK_H

#include <atomic>

namespace mindspore {
namespace kernel {
class SimpleSpinLock {
 public:
  void lock() {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { lock_.clear(std::memory_order_release); }

 private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_INTERNAL_SPINLOCK_H
