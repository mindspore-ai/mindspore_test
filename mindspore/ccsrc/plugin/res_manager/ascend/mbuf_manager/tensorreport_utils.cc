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
#include "plugin/res_manager/ascend/mbuf_manager/tensorreport_utils.h"
#include <dlfcn.h>
#include <libgen.h>
#include <memory>
#include <string>
#include <vector>
#include "utils/log_adapter.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "mindspore/ops/op_def/image_op_name.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_exception.h"

constexpr char kOptimizerEndFlag[] = "optimizer_end";

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

OptimizerEventInfo &OptimizerEventInfo::GetInstance() {
  static OptimizerEventInfo instance;
  return instance;
}

void OptimizerEventInfo::RecordEvent(bool is_optimizer_start, void *stream) {
  auto &opt_event = (is_optimizer_start ? optimizer_start_event_ : optimizer_end_event_);
  if (opt_event == nullptr) {
    if (aclrtCreateEventExWithFlag(&opt_event, ACL_EVENT_TIME_LINE) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Create event for uce " << (is_optimizer_start ? "start" : "end") << " timestamp failed.";
      return;
    } else {
      MS_LOG(INFO) << "Create event for uce" << (is_optimizer_start ? "start" : "end") << " timestamp successfully.";
    }
  }
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Call aclrtRecordEvent for optimizer " << (is_optimizer_start ? "start" : "end")
                                    << " opt_event=" << opt_event << ", addr_of_opt_event=" << &opt_event;
  (void)CALL_ASCEND_API(aclrtRecordEvent, opt_event, stream);
}

void OptimizerEventInfo::GetOptimizerTimestamp(bool is_optimizer_start) {
  auto &opt_event = (is_optimizer_start ? optimizer_start_event_ : optimizer_end_event_);
  uint64_t timestamp = 0;
  aclError ret_code = CALL_ASCEND_API(aclrtEventGetTimestamp, opt_event, &timestamp);
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Call aclrtEventGetTimestamp for optimizer "
                                    << (is_optimizer_start ? "start" : "end") << " ret_code=" << ret_code
                                    << ", timestamp=" << timestamp << ", opt_event=" << opt_event;
  if (ret_code == ACL_SUCCESS) {
    if (is_optimizer_start) {
      optimizer_start_timestamp_ = timestamp;
    } else {
      optimizer_end_timestamp_ = timestamp;
    }
  } else {
    MS_LOG(ERROR) << "Call aclrtEventGetTimestamp for optimizer " << (is_optimizer_start ? "start" : "end")
                  << " ret_code=" << ret_code << ".";
  }
}

bool OptimizerEventInfo::IsOptimizerStartKernelMod(kernel::KernelMod *kernel_mod, const CNodePtr &kernel) {
  if (optimizer_start_kernel_mod_ != nullptr) {
    return optimizer_start_kernel_mod_ == kernel_mod;
  }
  if (kernel_mod->kernel_name() != kTensorReport) {
    return false;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  if (!prim->HasAttr(kOptimizerEndFlag)) {
    optimizer_start_kernel_mod_ = kernel_mod;
    return true;
  }
  return false;
}

bool OptimizerEventInfo::IsOptimizerEndKernelMod(kernel::KernelMod *kernel_mod, const CNodePtr &kernel) {
  if (optimizer_end_kernel_mod_ != nullptr) {
    return optimizer_end_kernel_mod_ == kernel_mod;
  }
  if (kernel_mod->kernel_name() != kTensorReport) {
    return false;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  if (prim->HasAttr(kOptimizerEndFlag)) {
    optimizer_end_kernel_mod_ = kernel_mod;
    return true;
  }
  return false;
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
  constexpr std::string_view optARF = "ARF:1";
  if (!tftEnv.empty() && (tftEnv.find(optUCE) != std::string::npos || tftEnv.find(optTTP) != std::string::npos ||
                          tftEnv.find(optARF) != std::string::npos)) {
    return true;
  }
  return false;
}

TensorReportUtils::TensorReportUtils() {}

TensorReportUtils::~TensorReportUtils() {}

void TensorReportUtils::ReportReceiveData(const ScopeAclTdtDataset &dataset) {
  MS_LOG(DEBUG) << "Enter report recevice data.";
  if (UCEException::IsEnableUCE()) {
    OptimizerEventInfo::GetInstance().GetOptimizerTimestamp(true);
  }
  if (_optStart != nullptr) {
    auto ret = _optStart(-1);
    MS_LOG(INFO) << "Send start updating optimizer event to TFT. ret=" << ret;
  }
  MS_LOG(DEBUG) << "Finish report recevice data.";
}

void TensorReportUtils::SetTFTCallBack(const TFT_StartUpdatingOsFunObj &optStart) { _optStart = optStart; }

}  // namespace mindspore::device::ascend
