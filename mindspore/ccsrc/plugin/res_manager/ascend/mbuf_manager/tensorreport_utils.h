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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORREPORT_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORREPORT_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "common/kernel.h"
#include "plugin/res_manager/ascend/mbuf_manager/mbuf_receive_manager.h"
#include "utils/dlopen_macro.h"
#include "utils/ms_utils.h"
#include "plugin/res_manager/ascend/visible.h"

namespace mindspore::device::ascend {

ORIGIN_METHOD(TFT_StartUpdatingOs, int, int64_t);
const std::pair<string, string> tensorreport_mapping{"ms_tensor_report", "TensorReport"};

class ASCEND_RES_MANAGER_EXPORT OptimizerEventInfo {
 public:
  static OptimizerEventInfo &GetInstance();
  ~OptimizerEventInfo() = default;
  DISABLE_COPY_AND_ASSIGN(OptimizerEventInfo);

  void RecordEvent(bool is_optimizer_start, void *stream);

  void GetOptimizerTimestamp(bool is_optimizer_start);

  bool IsOptimizerStartKernelMod(kernel::KernelMod *kernel_mod, const CNodePtr &kernel);

  bool IsOptimizerEndKernelMod(kernel::KernelMod *kernel_mod, const CNodePtr &kernel);

  uint64_t get_optimizer_start_timestamp() { return optimizer_start_timestamp_; }

  uint64_t get_optimizer_end_timestamp() { return optimizer_end_timestamp_; }

  void Reset() {
    optimizer_start_event_ = nullptr;
    optimizer_end_event_ = nullptr;
    optimizer_start_timestamp_ = 0;
    optimizer_end_timestamp_ = 0;

    optimizer_start_kernel_mod_ = nullptr;
    optimizer_end_kernel_mod_ = nullptr;
  }

 private:
  OptimizerEventInfo() = default;

  aclrtEvent optimizer_start_event_ = nullptr;
  aclrtEvent optimizer_end_event_ = nullptr;
  uint64_t optimizer_start_timestamp_ = 0;
  uint64_t optimizer_end_timestamp_ = 0;

  // buffering kernel_mod pointers for speed up
  kernel::KernelMod *optimizer_start_kernel_mod_ = nullptr;
  kernel::KernelMod *optimizer_end_kernel_mod_ = nullptr;
};

class ASCEND_RES_MANAGER_EXPORT TensorReportUtils {
 public:
  static TensorReportUtils &GetInstance();

  ~TensorReportUtils();
  TensorReportUtils(const TensorReportUtils &) = delete;
  TensorReportUtils &operator=(const TensorReportUtils &) = delete;
  void ReportReceiveData(const ScopeAclTdtDataset &dataset);
  void SetTFTCallBack(const TFT_StartUpdatingOsFunObj &optStart);
  static bool IsEnable();

 private:
  // singleton instance, make constructor private
  TensorReportUtils();
  TFT_StartUpdatingOsFunObj _optStart = nullptr;
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORREPORT_UTILS_H_
