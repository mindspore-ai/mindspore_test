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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_THREAD_H
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_THREAD_H
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <signal.h>
#include <sys/prctl.h>
#include <pthread.h>
#endif
#include <string>

namespace mindspore {
namespace profiler {
namespace ascend {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
class Thread {
 public:
  Thread() : is_alive_(false), pid_(0), thread_name_("NPUProfiler") {}

  ~Thread() {
    if (is_alive_) {
      (void)pthread_cancel(pid_);
      (void)pthread_join(pid_, nullptr);
    }
  }

  void SetThreadName(const std::string &name) {
    if (!name.empty()) {
      thread_name_ = name;
    }
  }

  std::string GetThreadName() { return thread_name_; }

  int Start() {
    int ret = pthread_create(&pid_, nullptr, Execute, (void *)this);  // NOLINT(readability/casting)
    is_alive_ = (ret == 0) ? true : false;
    return ret;
  }

  int Stop() { return Join(); }

  int Join() {
    int ret = pthread_join(pid_, nullptr);
    is_alive_ = (ret == 0) ? false : true;
    return ret;
  }

 private:
  static void *Execute(void *args) {
    Thread *thr = (Thread *)args;                                                  // NOLINT(readability/casting)
    prctl(PR_SET_NAME, reinterpret_cast<uintptr_t>(thr->GetThreadName().data()));  // NOLINT(runtime/int)
    thr->Run();
    return nullptr;
  }
  virtual void Run() = 0;

 private:
  bool is_alive_;
  pthread_t pid_;
  std::string thread_name_;
};
#else
class Thread {
 public:
  Thread() {}
  ~Thread() {}
  void SetThreadName(const std::string &name) {}
  std::string GetThreadName() { return ""; }
  int Start() { return 0; }
  int Stop() { return 0; }
  int Join() { return 0; }
};
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_THREAD_H
