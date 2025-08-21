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
#include "acl_prof_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mindspore::device::ascend {
aclprofCreateConfigFunObj aclprofCreateConfig_ = nullptr;
aclprofDestroyConfigFunObj aclprofDestroyConfig_ = nullptr;
aclprofFinalizeFunObj aclprofFinalize_ = nullptr;
aclprofInitFunObj aclprofInit_ = nullptr;
aclprofStartFunObj aclprofStart_ = nullptr;
aclprofStopFunObj aclprofStop_ = nullptr;
aclprofCreateStepInfoFunObj aclprofCreateStepInfo_ = nullptr;
aclprofGetStepTimestampFunObj aclprofGetStepTimestamp_ = nullptr;
aclprofDestroyStepInfoFunObj aclprofDestroyStepInfo_ = nullptr;
aclprofGetSupportedFeaturesFunObj aclprofGetSupportedFeatures_ = nullptr;
aclprofGetSupportedFeaturesFunObj aclprofGetSupportedFeaturesV2_ = nullptr;

void LoadProfApiSymbol(const std::string &ascend_path) {
  std::string profiler_plugin_path = ascend_path + "lib64/libmsprofiler.so";
  auto handler = GetLibHandler(profiler_plugin_path, true);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << profiler_plugin_path << " failed!" << dlerror();
    return;
  }
  aclprofCreateConfig_ = DlsymAscendFuncObj(aclprofCreateConfig, handler);
  aclprofDestroyConfig_ = DlsymAscendFuncObj(aclprofDestroyConfig, handler);
  aclprofFinalize_ = DlsymAscendFuncObj(aclprofFinalize, handler);
  aclprofInit_ = DlsymAscendFuncObj(aclprofInit, handler);
  aclprofStart_ = DlsymAscendFuncObj(aclprofStart, handler);
  aclprofStop_ = DlsymAscendFuncObj(aclprofStop, handler);
  aclprofCreateStepInfo_ = DlsymAscendFuncObj(aclprofCreateStepInfo, handler);
  aclprofGetStepTimestamp_ = DlsymAscendFuncObj(aclprofGetStepTimestamp, handler);
  aclprofDestroyStepInfo_ = DlsymAscendFuncObj(aclprofDestroyStepInfo, handler);
  aclprofGetSupportedFeatures_ = DlsymAscendFuncObj(aclprofGetSupportedFeatures, handler);
  aclprofGetSupportedFeaturesV2_ = DlsymAscendFuncObj(aclprofGetSupportedFeaturesV2, handler);
  MS_LOG(INFO) << "Load acl prof api success!";
}

void LoadSimulationProfApi() {
  ASSIGN_SIMU(aclprofCreateConfig);
  ASSIGN_SIMU(aclprofDestroyConfig);
  ASSIGN_SIMU(aclprofFinalize);
  ASSIGN_SIMU(aclprofInit);
  ASSIGN_SIMU(aclprofStart);
  ASSIGN_SIMU(aclprofStop);
  ASSIGN_SIMU(aclprofCreateStepInfo);
  ASSIGN_SIMU(aclprofGetStepTimestamp);
  ASSIGN_SIMU(aclprofDestroyStepInfo);
  ASSIGN_SIMU(aclprofGetSupportedFeatures);
  ASSIGN_SIMU(aclprofGetSupportedFeaturesV2);
}
}  // namespace mindspore::device::ascend
