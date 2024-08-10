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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORREPORT_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORREPORT_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"
#include "utils/dlopen_macro.h"

namespace mindspore::device::ascend {

ORIGIN_METHOD(TFT_StartUpdatingOs, int, int64_t);
const std::pair<string, string> tensorreport_mapping{"ms_tensor_report", "TensorReport"};

class TensorReportUtils {
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
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORREPORT_UTILS_H_
