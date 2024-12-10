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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_LAZY_FUSION_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_LAZY_FUSION_KERNEL_H_

#include <queue>
#include <mutex>
#include <functional>
#include "include/backend/visible.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
class BACKEND_EXPORT LazyFusionKernel {
 public:
  LazyFusionKernel() = default;
  virtual ~LazyFusionKernel() = default;
  virtual void Flush() {}

  void Reset(const device::DeviceContext *context, size_t stream_id) {
    device_context_ = context;
    stream_id_ = stream_id;
  }
  const device::DeviceContext *device_context() const { return device_context_; }
  size_t stream_id() const { return stream_id_; }
  void set_id(size_t id) { id_ = id; }
  size_t id() const { return id_; }

 protected:
  const device::DeviceContext *device_context_;
  size_t stream_id_;
  size_t id_{0};
};

class BACKEND_EXPORT LazyFusionManager {
 public:
  using BuildFunc = std::function<LazyFusionKernel *()>;
  LazyFusionManager() = default;
  ~LazyFusionManager();

  void Register(const BuildFunc &builder) { build_func_ = builder; }

  LazyFusionKernel *Get(const device::DeviceContext *context, size_t stream) {
    if (current_) {
      if (current_->stream_id() == stream) {
        return current_;
      }
      current_->Flush();
    }
    current_ = NewKernel();
    current_->Reset(context, stream);
    current_->set_id(id_.fetch_add(1, std::memory_order_relaxed));
    return current_;
  }

  void Flush() {
    if (current_ != nullptr) {
      current_->Flush();
      current_ = nullptr;
    }
  }

  void FreeKernel(LazyFusionKernel *k) {
    std::lock_guard<std::mutex> guard(mutex_);
    pool_.push(k);
  }

 private:
  LazyFusionKernel *NewKernel() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!pool_.empty()) {
        auto k = pool_.front();
        pool_.pop();
        return k;
      }
    }
    return build_func_();
  }

  BuildFunc build_func_;
  std::queue<LazyFusionKernel *> pool_;
  std::mutex mutex_;
  LazyFusionKernel *current_{nullptr};
  std::atomic<size_t> id_{0};
};

extern BACKEND_EXPORT LazyFusionManager g_lazy_fusion_manager;

// before run/push current task, should generate dvm device task first
static inline void FlushLazyFusion() { g_lazy_fusion_manager.Flush(); }

template <typename T>
class LazyFusionRegister {
 public:
  LazyFusionRegister() {
    g_lazy_fusion_manager.Register([]() -> T * { return new T(); });
  }
  ~LazyFusionRegister() = default;
};

#define MS_REGISTER_LAZY_FUSION_KERNEL(CLASS_NAME) static const LazyFusionRegister<CLASS_NAME> g_lazy_fusion_reg;
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_LAZY_FUSION_KERNEL_H_
