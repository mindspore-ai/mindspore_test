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

#include "plugin/device/ascend/llm_boost/ascend_native_boost_builder.h"
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "common/ms_factory.h"
#include "utils/ms_utils.h"
#include "mindspore/ccsrc/backend/operator/boost_base_model.h"

namespace mindspore::kernel {
AscendNativeBoostBuilder::AscendNativeBoostBuilder() : BoostBaseBuilder("AscendNative") { Initialize(); }

AscendNativeBoostBuilder::~AscendNativeBoostBuilder() {}

void AscendNativeBoostBuilder::Initialize() {
  anb_lib_name_ = device::ascend::GetCurrentDir() + "/ascend/libms_ascend_native_boost.so";
#ifndef _WIN32
  lib_ptr_ = dlopen(anb_lib_name_.c_str(), RTLD_GLOBAL | RTLD_LAZY);
  std::string err_msg = GetDlErrorMsg();
#else
  lib_ptr_ = LoadLibrary(anb_lib_name_.c_str());
  std::string err_msg = std::to_string(GetLastError());
#endif
  if (lib_ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " << anb_lib_name_ << " failed. Error: " << err_msg;
  }
  model_func_ = (ModelFunc)dlsym(lib_ptr_, model_func_name_);
  err_msg = GetDlErrorMsg();
  if (model_func_ == nullptr) {
    MS_LOG(EXCEPTION) << "Loading model creator failed. Error: " << err_msg;
  }
}

void AscendNativeBoostBuilder::Finalize() {
  MS_EXCEPTION_IF_NULL(lib_ptr_);
#ifndef _WIN32
  if (dlclose(lib_ptr_) != 0) {
    MS_LOG(EXCEPTION) << "Closing " << anb_lib_name_ << " handle failed. Error: " << GetDlErrorMsg();
  }
#else
  if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(lib_ptr_))) {
    MS_LOG(EXCEPTION) << "Closing " << anb_lib_name_ << " handle failed. Error: " << std::to_string(GetLastError());
  }
#endif
}

std::shared_ptr<BoostBaseModel> AscendNativeBoostBuilder::BuildModel(const std::string &model_name) {
  MS_LOG(INFO) << "Create AscendNative TransformerBoost, model_name: " << model_name;
  return model_func_(model_name);
}

static std::shared_ptr<BoostBaseBuilder> CreateAscendNativeBoostBuilder() {
  return std::make_shared<AscendNativeBoostBuilder>();
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(BoostBaseBuilder, AscendNative, CreateAscendNativeBoostBuilder);

}  // namespace mindspore::kernel
