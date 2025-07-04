/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "common/ms_factory.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace embedding_look_up_cpu {
class EmbeddingLookUpCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<EmbeddingLookUpCpuKernelMod> {
 public:
  EmbeddingLookUpCpuKernelMod() = default;
  ~EmbeddingLookUpCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

  int64_t offset_;
  size_t input_indices_lens_{1};
  size_t first_dim_size_{1};
  size_t outer_dim_size_{1};
  TypeId input_indices_dtype_{kNumberTypeInt32};
  TypeId input_params_dtype_{kTypeUnknown};

  // This flag indicates whether the embedding storage capability is enabled, which supports hot data caching and
  // persistent storage of non-hotspot data for embedding tables, which is generally used in very large embedding table
  // scenarios.
  bool enable_embedding_storage_{false};
  // The global unique parameter key, used to get the embedding storage instance.
  int32_t parameter_key_{-1};
};
}  // namespace embedding_look_up_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
