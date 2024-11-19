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
#include "availability/silent_check/silent_check.h"
#include <map>
#include <memory>
#include <string>
#include "include/common/utils/stream_util.h"
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace silentcheck {
namespace {
constexpr char kDefaultSilentChecker[] = "SilentCheckerBase";
}  // namespace

std::map<std::string, std::shared_ptr<SilentCheckerBase>> &SilentCheckerBase::GetInstanceMap() {
  static std::map<std::string, std::shared_ptr<SilentCheckerBase>> instance_map = {};
  return instance_map;
}

std::shared_ptr<SilentCheckerBase> SilentCheckerBase::GetInstance() {
  auto &dev_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto iter = GetInstanceMap().find(dev_target);
  if (iter != GetInstanceMap().end()) {
    return iter->second;
  }
  return nullptr;
}

bool SilentCheckerBase::Register(const std::string &name, const std::shared_ptr<SilentCheckerBase> &instance) {
  if (GetInstanceMap().find(name) != GetInstanceMap().end()) {
    MS_LOG(WARNING) << "SilentChecker for target " << name << " has been registered.";
  } else {
    (void)GetInstanceMap().emplace(name, instance);
  }
  return true;
}

void SilentCheckerBase::ClearAll() {
  for (auto &elem : GetInstanceMap()) {
    elem.second->Clear();
  }
  // Note do not clear `GetInstanceMap()`, keep trainning after uce fault can still use silent check
}

bool SilentCheckerBase::NeedInsertCheckForLastGrad() {
  if (!IsNpuAsdEnable()) {
    return false;
  }
  // check whether is pipeline parallel stage 0
  return pp_stage_ == 0;
}
}  // namespace silentcheck
}  // namespace mindspore
