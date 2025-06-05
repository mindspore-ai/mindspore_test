/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/custom/custom_op_plugin_kernel.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <functional>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
CustomOpPluginCpuKernelMod::~CustomOpPluginCpuKernelMod() {
#if !defined(_WIN32) && !defined(_WIN64)
  if (handle_ != nullptr) {
    dlclose(handle_);
  }

#endif
}

void CustomOpPluginCpuKernelMod::SetKernelPath() {
  const char *op_plugin_path = common::EnvHelper::GetInstance()->GetEnv("MS_OP_PLUGIN_PATH");

  if (op_plugin_path == nullptr) {
    MS_LOG(EXCEPTION) << "Try to select kernel: " << primitive_->name() << ", but MS_OP_PLUGIN_PATH is not set";
  }

  auto real_path = FileUtils::GetRealPath(op_plugin_path);
  file_path_ = real_path.value();
  func_name_ = primitive_->name();
}

bool CustomOpPluginCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  kernel_name_ = primitive_->name();
  SetKernelPath();

  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    auto dtype = inputs[i]->dtype_id();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
    (void)type_list_.emplace_back(TypeIdToString(dtype, true));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    auto dtype = outputs[i]->dtype_id();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
    (void)type_list_.emplace_back(TypeIdToString(dtype, true));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  (void)std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                       [](auto &str) { return str.c_str(); });

  return true;
}

bool CustomOpPluginCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  std::vector<void *> params;
  kernel_info_.SetKernelInput(inputs);

  for (size_t i = 0; i < inputs.size(); i++) {
    params.push_back(static_cast<void *>(inputs[i]->device_ptr()));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    params.push_back(static_cast<void *>(outputs[i]->device_ptr()));
  }

  for (size_t i = 0; i < workspace.size(); i++) {
    params.push_back(static_cast<void *>(workspace[i]->device_ptr()));
  }

#if !defined(_WIN32) && !defined(_WIN64)

  if (!handle_) {
    handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on CPU, dlopen file '" << file_path_
                    << "' should be successful, but error occurs! Error message is: " << dlerror();
      return false;
    }
  }

  if (!aot_func_) {
    aot_func_ =
      reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
        dlsym(handle_, func_name_.c_str()));
    if (auto error_info = dlerror(); error_info != nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, error occurs when fetching function '" << func_name_
                        << "'. Error info: " << error_info;
    }
  }

  int nparam = SizeToInt(params.size());
  int ret = 0;
  try {
    if (nparam == 0) {
      ret = aot_func_(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      ret = aot_func_(nparam, &params[0], &ndims_[0], &shapes_[0], &type_pointer_list_[0], nullptr,
                      reinterpret_cast<void *>(&kernel_info_));
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, operator failed when executing user defined file "
                      << file_path_ << "! "
                      << "Error message is " << e.what();
  }

  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Return value from CPU AOT kernel(" << file_path_ << ")'s function(" << func_name_ << ") is "
                      << ret << ". "
                      << "Any return value not equal to 0 will be treated as user defined error code and we will "
                         "terminate execution. If termination is not your purpose, please set return value to 0.";
  }

#else
  MS_LOG(EXCEPTION) << "Custom AOT Operator doesn't support Windows currently";
#endif

  return true;
}

int CustomOpPluginCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  shapes_.clear();
  shape_list_.clear();
  ndims_.clear();

  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });

  return static_cast<int>(KRET_OK);
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CustomOpPlugin, CustomOpPluginCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
