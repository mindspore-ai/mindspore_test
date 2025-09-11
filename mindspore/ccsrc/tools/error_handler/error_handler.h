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

#ifndef MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
#define MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "include/backend/visible.h"
#include "include/backend/kernel_graph.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"

namespace mindspore {
namespace tools {
class BACKEND_COMMON_EXPORT ErrorHandler {
 public:
  static ErrorHandler &GetInstance();

  ErrorHandler() = default;
  virtual ~ErrorHandler() = default;
  // disable copy constructor and the assignment operator
  ErrorHandler(const ErrorHandler &) = delete;
  ErrorHandler &operator=(const ErrorHandler &) = delete;

  void SaveConstants(const std::vector<KernelGraphPtr> &graphs);
  const ValuePtr &GetConstant(const AnfNodePtr &node);
  void Clear();

 private:
  // save constant values for uce scenario, for constant tensor device memory may be corrupted
  std::map<AnfNodePtr, ValuePtr> const_values_{};
};

// Parameter snapshot manager
class BACKEND_COMMON_EXPORT SnapshotMgr {
 public:
  static std::shared_ptr<SnapshotMgr> GetInstance(const std::string &device);
  static std::map<std::string, std::shared_ptr<SnapshotMgr>> &GetInstanceMap();
  static bool Register(const std::string &name, const std::shared_ptr<SnapshotMgr> &instance);
  static void Clear();

  SnapshotMgr() = default;
  virtual ~SnapshotMgr() = default;
  // disable copy constructor and the assignment operator
  SnapshotMgr(const SnapshotMgr &) = delete;
  SnapshotMgr &operator=(const SnapshotMgr &) = delete;

  bool IsSavingSnapshot() const { return is_saving_snapshot_; }
  void SetSavingSnapshot(bool val) { is_saving_snapshot_ = val; }

  std::map<std::string, tensor::TensorPtr> &GetSavedParams() { return saved_params_; }

  int LastSaveStep() const { return last_save_step_; }
  void SaveLastSaveStep(int val) { last_save_step_ = val; }

  bool IsSnapshotValid() { return last_save_step_ > 0; }

  void Reset() {
    last_save_step_ = 0;
    saved_params_.clear();
  }

 protected:
  // whether is in the progress of copying parameters from device to host
  bool is_saving_snapshot_ = false;

  std::map<std::string, tensor::TensorPtr> saved_params_;
  int last_save_step_ = 0;
};

using SnapshotMgrPtr = std::shared_ptr<SnapshotMgr>;
}  // namespace tools
}  // namespace mindspore

#define SNAPSHOT_MANAGER_REG(NAME, CLAZZ)         \
  static bool g_SnapshotMgr_##NAME##_reg_result = \
    mindspore::tools::SnapshotMgr::Register(NAME, std::make_shared<CLAZZ>())

#endif  // MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
