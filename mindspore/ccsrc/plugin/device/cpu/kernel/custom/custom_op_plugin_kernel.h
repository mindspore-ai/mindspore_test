/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/custom/custom_kernel_input_info.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {

class CustomOpPluginCpuKernelMod : public NativeCpuKernelMod {
 public:
  CustomOpPluginCpuKernelMod() : handle_(nullptr), resize_func_(nullptr), aot_func_(nullptr) {}
  ~CustomOpPluginCpuKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<int> ndims_;
  std::vector<std::string> type_list_;

  std::vector<int64_t *> shapes_;
  std::vector<const char *> type_pointer_list_;

  std::string file_path_;
  std::string func_name_;
  void *handle_{nullptr};
  // int (*init_func_)(int *, int64_t **, const char **, KernelInputInfo *);
  int (*resize_func_)(int *, int64_t **, const char **, KernelInputInfo *);
  int (*aot_func_)(int, void **, int *, int64_t **, const char **, void *, void *);

  KernelInputInfoImpl kernel_info_;

 private:
  void SetKernelPath();
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_
