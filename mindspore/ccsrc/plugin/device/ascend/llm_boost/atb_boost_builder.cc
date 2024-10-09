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

#include "plugin/device/ascend/llm_boost/atb_boost_builder.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/factory/ms_factory.h"
#include "utils/ms_utils.h"
#include "include/common/debug/common.h"

namespace mindspore::kernel {
AtbBoostBuilder::AtbBoostBuilder() : BoostBaseBuilder("DEFAULT") { Initialize(); }

AtbBoostBuilder::~AtbBoostBuilder() {}

void AtbBoostBuilder::Initialize() {
  // load atb adapter so when called atb boost
  std::string err_msg = "";
  std::string atb_model_path_env = common::GetEnv("ATB_MODEL_HOME_PATH");
  if (atb_model_path_env == "") {
    MS_LOG(EXCEPTION) << "ATB_MODEL_HOME_PATH is not set";
  }
  const std::string atb_model_path = atb_model_path_env;
  std::string atb_model_libs_path = atb_model_path + "/lib";

  std::string atb_core_lib_path = atb_model_libs_path + "/" + "libatb_speed_core.so";
  auto core_lib_ptr = dlopen(atb_core_lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  err_msg = GetDlErrorMsg();
  if (core_lib_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + std::string("libatb_speed_core.so") + " failed. Error: " + err_msg;
  }

  std::string atb_layer_lib_path = atb_model_libs_path + "/" + "libatb_speed_layers.so";
  std::string atb_operation_lib_path = atb_model_libs_path + "/" + "libatb_speed_operations.so";
  if (!Common::FileExists(atb_layer_lib_path) && !Common::FileExists(atb_operation_lib_path)) {
    MS_LOG(EXCEPTION) << std::string("libatb_speed_layers.so") + " or " + std::string("libatb_speed_operations.so") +
                           " not found";
  }

  // Compatible with older versions
  if (Common::FileExists(atb_layer_lib_path)) {
    auto layer_lib_ptr = dlopen(atb_layer_lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    err_msg = GetDlErrorMsg();
    if (layer_lib_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Loading " + std::string("libatb_speed_layers.so") + " failed. Error: " + err_msg;
    }
  }

  if (Common::FileExists(atb_operation_lib_path)) {
    auto operation_lib_ptr = dlopen(atb_operation_lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    err_msg = GetDlErrorMsg();
    if (operation_lib_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Loading " + std::string("libatb_speed_operations.so") + " failed. Error: " + err_msg;
    }
  }

  std::string atb_model_lib_path = atb_model_libs_path + "/" + "libatb_speed_models.so";
  auto model_lib_ptr = dlopen(atb_model_lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  err_msg = GetDlErrorMsg();
  if (model_lib_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + std::string("libatb_speed_models.so") + " failed. Error: " + err_msg;
  }

  atb_boost_lib_name_ = device::ascend::GetCurrentDir() + "/ascend/libms_atb_boost.so";
#ifndef _WIN32
  lib_ptr_ = dlopen(atb_boost_lib_name_.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#else
  lib_ptr_ = LoadLibrary(atb_boost_lib_name_.c_str());
  err_msg = std::to_string(GetLastError());
#endif
  if (lib_ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + atb_boost_lib_name_ + " failed. Error: " + err_msg;
  }
  model_func_ = (ModelFunc)dlsym(lib_ptr_, model_func_name_);
  err_msg = GetDlErrorMsg();
  if (model_func_ == nullptr) {
    MS_LOG(EXCEPTION) << "Loading model creator failed. Error: " + err_msg;
  }
}

void AtbBoostBuilder::Finalize() {
  MS_EXCEPTION_IF_NULL(lib_ptr_);

#ifndef _WIN32
  if (dlclose(lib_ptr_) != 0) {
    MS_LOG(EXCEPTION) << "Closing " + atb_boost_lib_name_ + " handle failed. Error: " + GetDlErrorMsg();
  }
#else
  if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(lib_ptr_))) {
    MS_LOG(EXCEPTION) << "Closing " + atb_boost_lib_name_ + " handle failed. Error: " + std::to_string(GetLastError());
  }
#endif
}

std::shared_ptr<BoostBaseModel> AtbBoostBuilder::BuildModel(const std::string &model_name) {
  return model_func_(model_name);
}

static std::shared_ptr<BoostBaseBuilder> CreateAtbBoostBuilder() { return std::make_shared<AtbBoostBuilder>(); }

MS_KERNEL_FACTORY_REG_BY_CREATOR(BoostBaseBuilder, ATB, CreateAtbBoostBuilder);
}  // namespace mindspore::kernel
