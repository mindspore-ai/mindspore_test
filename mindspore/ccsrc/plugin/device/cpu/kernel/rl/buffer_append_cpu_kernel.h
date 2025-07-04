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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_APPEND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_APPEND_CPU_KERNEL_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
class BufferAppendCpuKernelMod : public NativeCpuKernelMod {
 public:
  BufferAppendCpuKernelMod() : element_nums_(0), exp_batch_(0), capacity_(0) {}

  ~BufferAppendCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto shapes = GetValue<std::vector<int64_t>>(primitive_->GetAttr("buffer_elements"));
    auto types = GetValue<std::vector<TypePtr>>(primitive_->GetAttr("buffer_dtype"));
    capacity_ = GetValue<int64_t>(primitive_->GetAttr("capacity"));
    exp_batch_ = GetValue<int64_t>(primitive_->GetAttr("exp_batch"));
    // check capacity > 0
    if (capacity_ <= 0) {
      MS_LOG(EXCEPTION) << "Capacity must be greater than 0 ";
    }
    element_nums_ = shapes.size();
    for (size_t i = 0; i < element_nums_; i++) {
      exp_element_list.push_back(LongToSize(shapes[i]) * UnitSizeInBytes(types[i]->type_id()));
    }
    output_size_list_.clear();
    output_size_list_.push_back(sizeof(int));
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &) override {
    auto count_addr = GetDeviceAddress<int>(inputs, kSecondInputIndex * element_nums_);
    auto head_addr = GetDeviceAddress<int>(inputs, kSecondInputIndex * element_nums_ + 1);
    MS_EXCEPTION_IF_NULL(count_addr);
    MS_EXCEPTION_IF_NULL(head_addr);
    int index = 0;
    if (count_addr[0] <= capacity_ - 1 && head_addr[0] == 0) {
      index = count_addr[0];
      count_addr[0] = index + LongToInt(exp_batch_);
      if (count_addr[0] > capacity_) {
        count_addr[0] = LongToInt(capacity_);
        head_addr[0] = (LongToInt(exp_batch_) + count_addr[0] - LongToInt(capacity_)) % LongToInt(capacity_);
      }
    } else {
      index = head_addr[0];
      head_addr[0] = (LongToInt(exp_batch_) + head_addr[0]) % LongToInt(capacity_);
    }
    // If exp_batch > (capcity_ - index), goto buffer's head
    int remain_size = (exp_batch_ > (capacity_ - index)) ? LongToInt(capacity_ - index) : LongToInt(exp_batch_);
    int remap_size = (exp_batch_ > (capacity_ - index)) ? LongToInt(exp_batch_ - (capacity_ - index)) : 0;
    auto task = [this, &inputs, index, remain_size, remap_size](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
        auto exp_addr = GetDeviceAddress<unsigned char>(inputs, i + element_nums_);
        MS_EXCEPTION_IF_NULL(buffer_addr);
        MS_EXCEPTION_IF_NULL(exp_addr);
        size_t one_exp_len = exp_element_list[i];
        size_t dist_len = one_exp_len;
        if (memcpy_s(buffer_addr + IntToSize(index) * one_exp_len, one_exp_len * IntToSize(remain_size), exp_addr,
                     dist_len * IntToSize(remain_size)) != EOK) {
          MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed";
        }
        if (remap_size > 0) {
          if (memcpy_s(buffer_addr, one_exp_len * IntToSize(remap_size), exp_addr, dist_len * IntToSize(remap_size)) !=
              EOK) {
            MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed";
          }
        }
      }
    };
    ParallelLaunchAutoSearch(task, element_nums_, this, &parallel_search_info_);
    return true;
  }

 private:
  size_t element_nums_;
  int64_t exp_batch_;
  int64_t capacity_;
  std::vector<size_t> exp_element_list;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_APPEND_CPU_KERNEL_H_
