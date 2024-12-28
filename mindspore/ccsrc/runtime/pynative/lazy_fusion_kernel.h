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

#include <map>
#include <queue>
#include <mutex>
#include <string>
#include <utility>
#include <functional>
#include "include/backend/visible.h"
#include "runtime/hardware/device_context.h"
#include "runtime/runtime_conf/runtime_conf.h"
#include "runtime/pynative/lazy_fusion_flags.h"

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

using LazyFusionBuildFunc = std::function<LazyFusionKernel *()>;
using LazyFusionInitFunc = std::function<void()>;

class BACKEND_EXPORT LazyFusionManager {
 public:
  LazyFusionManager() = default;
  ~LazyFusionManager();

  void Register(const std::string &device_name, const LazyFusionBuildFunc &func) { build_funcs_[device_name] = func; }
  void RegisterInit(const std::string &device_name, const LazyFusionInitFunc &func) { init_funcs_[device_name] = func; }

  void Init();

  LazyFusionKernel *Get(const device::DeviceContext *context, size_t stream) {
    if (current_ != nullptr) {
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

  std::map<std::string, LazyFusionBuildFunc> build_funcs_;
  std::map<std::string, LazyFusionInitFunc> init_funcs_;
  LazyFusionBuildFunc build_func_{nullptr};
  LazyFusionInitFunc init_func_{nullptr};
  std::queue<LazyFusionKernel *> pool_;
  std::mutex mutex_;
  LazyFusionKernel *current_{nullptr};
  std::atomic<size_t> id_{0};
};

extern BACKEND_EXPORT LazyFusionManager g_lazy_fusion_manager;

// before run/push current task, should generate dvm device task first
static inline void FlushLazyFusion() { g_lazy_fusion_manager.Flush(); }
static inline void LazyFusionInit() {
  if (LazyFusionFlags::GetInstance().opt_level < OptLevel_1 || runtime::RuntimeConf::GetInstance()->launch_blocking() ||
      MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON") {
    return;
  }
  g_lazy_fusion_manager.Init();
}

template <typename T>
class LazyFusionRegister {
 public:
  explicit LazyFusionRegister(const std::string &device_name) {
    g_lazy_fusion_manager.Register(device_name, []() -> T * { return new T(); });
  }
  ~LazyFusionRegister() = default;
};

#define MS_REGISTER_LAZY_FUSION_KERNEL(DEVICE, CLASS_NAME) \
  static const LazyFusionRegister<CLASS_NAME> g_lazy_fusion_##DEVICE##_build_reg(DEVICE)

class LazyFusionInitRegister {
 public:
  LazyFusionInitRegister(const std::string &device_name, LazyFusionInitFunc &&func) {
    g_lazy_fusion_manager.RegisterInit(device_name, std::move(func));
  }
  ~LazyFusionInitRegister() = default;
};

#define MS_REGISTER_LAZY_FUSION_INIT(DEVICE, FUNC) \
  static const LazyFusionInitRegister g_lazy_fusion_##DEVICE##_int_reg(DEVICE, FUNC)
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_LAZY_FUSION_KERNEL_H_
