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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
class BufferGetCpuKernelMod : public NativeCpuKernelMod {
 public:
  BufferGetCpuKernelMod() : element_nums_(0) {}

  ~BufferGetCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto shapes = GetValue<std::vector<int64_t>>(primitive_->GetAttr("buffer_elements"));
    auto types = GetValue<std::vector<TypePtr>>(primitive_->GetAttr("buffer_dtype"));
    element_nums_ = shapes.size();
    for (size_t i = 0; i < element_nums_; i++) {
      exp_element_list.push_back(LongToSize(shapes[i]) * UnitSizeInBytes(types[i]->type_id()));
    }
    output_size_list_.clear();
    for (auto i : exp_element_list) {
      output_size_list_.push_back(i);
    }
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    auto count_addr = GetDeviceAddress<int>(inputs, element_nums_);
    auto head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
    auto index_addr = GetDeviceAddress<int>(inputs, element_nums_ + kSecondInputIndex);
    MS_EXCEPTION_IF_NULL(count_addr);
    MS_EXCEPTION_IF_NULL(head_addr);
    MS_EXCEPTION_IF_NULL(index_addr);
    int index = index_addr[0];
    if (index_addr[0] < 0) {
      index += count_addr[0];
    }
    if (!(index >= 0 && index < count_addr[0])) {
      MS_LOG(ERROR) << "The index " << index_addr[0] << " is out of range:[ " << -1 * count_addr[0] << ", "
                    << count_addr[0] << ").";
    }
    int t = count_addr[0] - head_addr[0];
    if (index < t) {
      index += head_addr[0];
    } else {
      index -= t;
    }
    auto task = [this, &inputs, &outputs, index](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
        auto item_addr = GetDeviceAddress<unsigned char>(outputs, i);
        MS_EXCEPTION_IF_NULL(buffer_addr);
        MS_EXCEPTION_IF_NULL(item_addr);
        size_t one_exp_len = output_size_list_[i];
        size_t dist_len = one_exp_len;
        if (memcpy_s(item_addr, one_exp_len, buffer_addr + IntToSize(index) * one_exp_len, dist_len) != EOK) {
          MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed";
        }
      }
    };
    ParallelLaunchAutoSearch(task, element_nums_, this, &parallel_search_info_);
    return true;
  }

 private:
  size_t element_nums_;
  std::vector<size_t> exp_element_list;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
