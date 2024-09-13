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
#include "plugin/device/ascend/hal/device/tensorreport_utils.h"
#include <dlfcn.h>
#include <libgen.h>
#include <memory>
#include <string>
#include <vector>
#include "ir/tensor.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::device::ascend {

static std::string GetCurDir() {
#ifndef _WIN32
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void *>(GetCurDir), &dlInfo) == 0) {
    MS_LOG(WARNING) << "GetCurDir fetch dladdr error.";
    return "";
  }
  std::string curSoPath(dlInfo.dli_fname);
  auto curDir = curSoPath.substr(0, curSoPath.find_last_of('/') + 1);
  MS_LOG(INFO) << "Get cur so dir is " << curDir;
  return curDir;
#else
  return "";
#endif
}

TensorReportUtils &TensorReportUtils::GetInstance() {
  static TensorReportUtils instance;
  static std::once_flag instInitFlag = {};
  std::call_once(instInitFlag, [&]() {
    auto curDir = GetCurDir();
    const std::string &msPrefix = "mindspore/lib/plugin";
    auto found = curDir.find(msPrefix);
    if (found != std::string::npos) {
      auto commPrefix = curDir.substr(0, found);
      const std::string &tftMsPrefix = commPrefix + "mindio_ttp/mindspore_api/";
      const std::string &tftCommPrefix = commPrefix + "mindio_ttp/framework_ttp/";
      const std::vector<string> depLibs = {"libttp_framework.so"};
      for (auto lPath : depLibs) {
        auto libPath = tftCommPrefix + lPath;
        void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
        if (!handle) {
          MS_LOG(WARNING) << "MindIO feature is switched on, but can't find MindIO install library: " << libPath
                          << "; Please check if MindIO package installed correctly!";
          return;
        }
      }
      auto tftSoPath = tftMsPrefix + "libttp_c_api.so";
      void *handle = dlopen(tftSoPath.c_str(), RTLD_LAZY);
      MS_LOG(DEBUG) << "Start dlopen TFT so path." << tftSoPath;
      if (handle) {
        MS_LOG(INFO) << "dlopen TFT so path successful." << tftSoPath;
        auto startFunc = DlsymWithCast<TFT_StartUpdatingOsFunPtr>(handle, "MindioTtpSetOptimStatusUpdating");
        if (startFunc) {
          MS_LOG(INFO) << "Found TFT optimizer status updating function.";
          instance.SetTFTCallBack(startFunc);
        } else {
          MS_LOG(WARNING) << "MindIO feature is switched on, but can't find report function: "
                             "MindioTtpSetOptimStatusUpdating; Please check if MindIO package installed correctly!";
        }
      }
    }
  });
  return instance;
}

bool TensorReportUtils::IsEnable() {
  auto tftEnv = common::GetEnv("MS_ENABLE_TFT");
  constexpr std::string_view optUCE = "UCE:1";
  constexpr std::string_view optTTP = "TTP:1";
  if (!tftEnv.empty() && (tftEnv.find(optUCE) != std::string::npos || tftEnv.find(optTTP) != std::string::npos)) {
    return true;
  }
  return false;
}

TensorReportUtils::TensorReportUtils() {}

TensorReportUtils::~TensorReportUtils() {}

void TensorReportUtils::ReportReceiveData(const ScopeAclTdtDataset &dataset) {
  MS_LOG(DEBUG) << "Enter report recevice data.";
  if (_optStart != nullptr) {
    auto ret = _optStart(-1);
    MS_LOG(INFO) << "Send start updating optimizer event to TFT. ret=" << ret;
  }
  MS_LOG(DEBUG) << "Finish report recevice data.";
}

void TensorReportUtils::SetTFTCallBack(const TFT_StartUpdatingOsFunObj &optStart) { _optStart = optStart; }

}  // namespace mindspore::device::ascend
