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

#include "include/backend/debug/tft_adapter/tft_wait_sem.h"
#include <mutex>
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace debug {
namespace tft {
TFTWaitSem &TFTWaitSem::GetInstance() {
  static TFTWaitSem instance;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  static std::once_flag initFlag = {};
  std::call_once(initFlag, [&]() { sem_init(&(instance.waitSem_), 0, 0); });
#endif
  return instance;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void TFTWaitSem::Wait() { sem_wait(&(GetInstance().waitSem_)); }
void TFTWaitSem::Post() { sem_post(&(GetInstance().waitSem_)); }
void TFTWaitSem::Clear() { sem_destroy(&(GetInstance().waitSem_)); }
#else
void TFTWaitSem::Wait() {}
void TFTWaitSem::Post() {}
void TFTWaitSem::Clear() {}
#endif
TFTWaitSem::TFTWaitSem() {}
TFTWaitSem::~TFTWaitSem() {}
bool TFTWaitSem::IsEnable() {
  auto msContext = MsContext::GetInstance();
  if (msContext->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode ||
      msContext->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    return false;
  }
  auto tftEnv = common::GetEnv("MS_ENABLE_TFT");
  constexpr std::string_view optUCE = "UCE:1";
  constexpr std::string_view optTTP = "TTP:1";
  if (!tftEnv.empty() && (tftEnv.find(optUCE) != std::string::npos || tftEnv.find(optTTP) != std::string::npos)) {
    return true;
  }
  return false;
}
}  // namespace tft
}  // namespace debug
}  // namespace mindspore
