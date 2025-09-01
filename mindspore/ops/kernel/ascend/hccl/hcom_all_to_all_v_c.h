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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_TO_ALL_V_C_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_TO_ALL_V_C_H_

#include <memory>
#include <vector>
#include <string>
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/hccl_kernel.h"
#include "kernel/ascend/hccl/hcom_util.h"

namespace mindspore {
namespace kernel {
class HcomAlltoAllVCKernel : public HcclKernel {
 public:
  HcomAlltoAllVCKernel() = default;
  ~HcomAlltoAllVCKernel() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int64_t GetOutputNumel(int64_t block_size, const std::vector<int64_t> &list_1d);

 protected:
  HcclDataType data_type_ = {};
  std::shared_ptr<int64_t[]> params_;
  int64_t rank_id_{0};
  int64_t rank_size_{0};
};

MS_HCCL_REG_KERNEL(AlltoAllVC, HcomAlltoAllVCKernel);
}  // namespace kernel
}  // namespace mindspore
#endif
