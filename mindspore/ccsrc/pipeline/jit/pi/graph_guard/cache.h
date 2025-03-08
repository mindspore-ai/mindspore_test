/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_CACHE_H
#define MINDSPORE_PI_JIT_CACHE_H

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/graph_guard/guard.h"
#include "pipeline/jit/pi/graph_guard/perf.h"

namespace mindspore {
namespace pijit {
using NativeFunc = std::function<PyObject *(PyObject *, PyObject *)>;
using ReleaseFunc = std::function<void()>;
class OptFunc {
 public:
  OptFunc(NativeFunc cFunc, ReleaseFunc rFunc);
  virtual ~OptFunc();
  NativeFunc GetFunc();

 protected:
  NativeFunc cFunc_;
  ReleaseFunc rFunc_;
};
using OptFuncPtr = std::shared_ptr<OptFunc>;

/// \brief OptOption is the compilation option for the code
class OptOption : public std::enable_shared_from_this<OptOption> {
 public:
  struct Less {
    bool operator()(const std::shared_ptr<OptOption> &a, const std::shared_ptr<OptOption> &b) const {
      return std::less<void *>()(a->target_, b->target_);
    }
  };

  /// \brief no support for default construction and you can extend the option class to support more feature
  OptOption() = delete;
  virtual ~OptOption() = default;
  /// \brief support create option by PyCodeObject
  static std::shared_ptr<OptOption> CreateOptionByCode(PyCodeObject *code);
  static std::shared_ptr<OptOption> CreateOptionByPoint(void *ptr);
  bool operator==(const OptOption &obj) const;

 protected:
  explicit OptOption(PyCodeObject *code);
  explicit OptOption(void *ptr);
  void *target_;
};
using OptOptionPtr = std::shared_ptr<OptOption>;

class GuardStatus {
 public:
  virtual ~GuardStatus() = default;
  virtual bool is_definitions_map() const { return false; }
};

/// \brief optimized code with native function graph and guard based on the compilation option
class OptCode : public std::enable_shared_from_this<OptCode> {
 public:
  OptCode();
  virtual ~OptCode();
  virtual void SetGuard(OptGuardPtr guard);
  virtual OptGuardPtr GetGuard();
  virtual void SetOption(OptOptionPtr option);
  virtual OptOptionPtr GetOption();
  virtual OptPerfPtr GetPerf(OptPerf::PerfKind kind);

  void SetPythonCode(const py::object &code);
  PyCodeObject *GetPythonCode() const;
  void SetNativeFunc(const std::string &phase, NativeFunc cFunc, ReleaseFunc rFunc);
  NativeFunc GetNativeFunc() const;
  std::string GetPhase() const;
  void Copy(std::shared_ptr<OptCode> dst);
  void Inc();
  uint64_t Count();
  std::shared_ptr<GuardStatus> &guard_status() { return guard_status_; }

 protected:
  std::string phase_;
  OptFuncPtr compiled_func_;
  py::object compiled_code_;
  OptGuardPtr guard_;
  OptOptionPtr option_;
  OptPerfPtr graph_perf_;
  OptPerfPtr pynative_perf_;
  uint64_t call_count_;
  std::shared_ptr<GuardStatus> guard_status_;
};
using OptCodePtr = std::shared_ptr<OptCode>;
using OptCodeSet = std::vector<OptCodePtr>;

using OptCodeFilterFunc = std::function<bool(OptCodePtr)>;
/// \brief hub for optimized code based on compilation option
class OptCodeHub : public std::enable_shared_from_this<OptCodeHub> {
 public:
  friend class CodeCache;
  OptCodeHub() = default;
  virtual ~OptCodeHub() = default;
  virtual OptCodePtr AddOptTarget(OptOptionPtr option);
  virtual const OptCodeSet &GetOptTarget(const OptOptionPtr &option, const OptCodeSet &defaults);
  virtual void UpdateOptTarget(OptOptionPtr option, OptCodePtr code);
  virtual void DelOptTarget(OptOptionPtr option, OptCodePtr code);
  virtual void DelOptTarget(OptCodePtr code);
  virtual std::vector<OptCodeSet> GetAllOptTarget();
  static void Register(std::string key, OptCodePtr code);
  static OptCodePtr Filter(std::string key, OptCodeFilterFunc filter);

 protected:
  // use OptOption instead of OptOptionPtr ...
  std::map<OptOptionPtr, OptCodeSet, OptOption::Less> codeMap_;
};

using OptCodeHubPtr = std::shared_ptr<OptCodeHub>;

class CodeCache {
 public:
  struct FailInfo {
    int count_;
  };

  explicit CodeCache(void *jcr);
  const auto &code_hub() const { return code_hub_; }
  const auto &fail_guard() const { return fail_guard_; }
  const auto &code() const { return code_; }

  void set_code(const OptCodePtr &ptr) { code_ = ptr; }
  FailInfo FindFailInfo(const GuardItemPtr &p) const;
  void CollectFailGuard();
  void Clear();

 private:
  class GuardItemKey {
   public:
    explicit GuardItemKey(const GuardItemPtr &p) : ptr_(p) {}
    auto operator-> () const { return ptr_.operator->(); }
    auto ptr() const { return ptr_; }
    size_t hash() const { return ptr_ ? ptr_->GetTrace()->Info().Id() + ptr_->GetType() : 0; }
    bool operator==(const GuardItemKey &) const noexcept;

   private:
    GuardItemPtr ptr_;
  };
  struct KeyHash {
    bool operator()(const GuardItemKey &p) const noexcept { return p.hash(); }
  };

  using GuardItemSet = std::unordered_set<GuardItemKey, KeyHash>;
  using FailGuardItemMap = std::unordered_map<GuardItemKey, FailInfo, KeyHash>;

  OptOptionPtr jcr_;
  OptCodeHubPtr code_hub_;
  OptCodePtr code_;
  FailGuardItemMap fail_guard_;  // total count same as compile time ...
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_CACHE_H
