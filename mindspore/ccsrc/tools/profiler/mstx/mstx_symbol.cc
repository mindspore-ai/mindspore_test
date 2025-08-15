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
#include "tools/profiler/mstx/mstx_symbol.h"
#include "tools/profiler/utils.h"

namespace mindspore {
namespace profiler {
bool g_mstx_support = false;
bool g_mstx_domain_support = false;

mstxMarkAFunObj mstxMarkA_ = nullptr;
mstxRangeStartAFunObj mstxRangeStartA_ = nullptr;
mstxRangeEndFunObj mstxRangeEnd_ = nullptr;
mstxDomainCreateAFunObj mstxDomainCreateA_ = nullptr;
mstxDomainDestroyFunObj mstxDomainDestroy_ = nullptr;
mstxDomainMarkAFunObj mstxDomainMarkA_ = nullptr;
mstxDomainRangeStartAFunObj mstxDomainRangeStartA_ = nullptr;
mstxDomainRangeEndFunObj mstxDomainRangeEnd_ = nullptr;
mstxMemRegionsRegisterFunObj mstxMemRegionsRegister_ = nullptr;
mstxMemRegionsUnregisterFunObj mstxMemRegionsUnregister_ = nullptr;
mstxMemHeapRegisterFunObj mstxMemHeapRegister_ = nullptr;

void LoadMstxApiSymbol(const std::string &ascend_path) {
  std::string mstx_plugin_path = ascend_path + "/lib64/libms_tools_ext.so";
  if (mindspore::profiler::Utils::RealPath(mstx_plugin_path).empty()) {
    MS_LOG(WARNING) << "Current cann does not support mstx.so!";
    return;
  }
#ifdef __linux__
  auto handler = mindspore::profiler::Utils::GetLibHandler(mstx_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen libms_tools_ext.so failed! Ignore this log if you don't use mstx.";
    return;
  }
  mstxMarkA_ = DlsymAscendFuncObj(mstxMarkA, handler);
  mstxRangeStartA_ = DlsymAscendFuncObj(mstxRangeStartA, handler);
  mstxRangeEnd_ = DlsymAscendFuncObj(mstxRangeEnd, handler);
  mstxDomainCreateA_ = DlsymAscendFuncObj(mstxDomainCreateA, handler);
  mstxDomainDestroy_ = DlsymAscendFuncObj(mstxDomainDestroy, handler);
  mstxDomainMarkA_ = DlsymAscendFuncObj(mstxDomainMarkA, handler);
  mstxDomainRangeStartA_ = DlsymAscendFuncObj(mstxDomainRangeStartA, handler);
  mstxDomainRangeEnd_ = DlsymAscendFuncObj(mstxDomainRangeEnd, handler);
  mstxMemRegionsRegister_ = DlsymAscendFuncObj(mstxMemRegionsRegister, handler);
  mstxMemRegionsUnregister_ = DlsymAscendFuncObj(mstxMemRegionsUnregister, handler);
  mstxMemHeapRegister_ = DlsymAscendFuncObj(mstxMemHeapRegister, handler);
#endif
  g_mstx_support = mstxMarkA_ != nullptr && mstxRangeStartA_ != nullptr && mstxRangeEnd_ != nullptr;
  g_mstx_domain_support = mstxDomainCreateA_ != nullptr && mstxDomainDestroy_ != nullptr &&
                          mstxDomainMarkA_ != nullptr && mstxDomainRangeStartA_ != nullptr &&
                          mstxDomainRangeEnd_ != nullptr && mstxMemRegionsRegister_ != nullptr &&
                          mstxMemRegionsUnregister_ != nullptr && mstxMemHeapRegister_ != nullptr;
  MS_LOG(INFO) << "Load mstx api success!";
}

bool IsCannSupportMstxApi() { return g_mstx_support; }

bool IsCannSupportMstxDomainApi() { return g_mstx_domain_support; }

}  // namespace profiler
}  // namespace mindspore
