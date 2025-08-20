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
#include "plugin/ascend/res_manager/symbol_interface/hal_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr auto kAscendHalSoPath = "lib64/driver/libascend_hal.so";

halHostRegisterFunObj halHostRegister_;
halHostUnregisterFunObj halHostUnregister_;

void LoadHalApiSymbol(const std::string &driver_path) {
  std::string ascend_hal_plugin_path = driver_path + kAscendHalSoPath;
  auto base_handler = GetLibHandler(ascend_hal_plugin_path);
  if (base_handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << ascend_hal_plugin_path << " failed!" << dlerror();
    return;
  }
  halHostRegister_ = DlsymAscendFuncObj(halHostRegister, base_handler);
  halHostUnregister_ = DlsymAscendFuncObj(halHostUnregister, base_handler);
  MS_LOG(INFO) << "Load ascend hal api success!";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
