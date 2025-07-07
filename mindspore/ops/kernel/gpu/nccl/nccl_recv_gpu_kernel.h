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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <utility>
#include "kernel/gpu/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace kernel {
class NcclRecvGpuKernel : public NcclGpuKernelMod, public MatchKernelHelper<NcclRecvGpuKernel> {
 public:
  NcclRecvGpuKernel() : src_rank_(-1) {}
  ~NcclRecvGpuKernel() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    size_t output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    unit_size_ = abstract::TypeIdSize(outputs[kIndex0]->dtype_id());
    SelectCollectiveHandle();
    return MatchKernelFunc(kernel_name_, inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    src_rank_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("src_rank")));
    group_name_ = GetValue<std::string>(primitive_->GetAttr(kAttrGroup));
    nccl_data_type_ = nccl_dtype(outputs[0]->dtype_id());

    auto shape_signed = outputs[0]->GetDeviceShapeVector();
    if (IsDynamic(shape_signed)) {
      return KRET_UNKNOWN_OUT_SHAPE;
    }
    output_size_list_.clear();
    auto output_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      return true;
    }
    size_t output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), unit_size_, std::multiplies<size_t>());
    output_size_list_.push_back(output_size);
    MS_LOG(INFO) << "NcclRecv source rank is " << src_rank_ << ", group name is " << group_name_;

    SelectCollectiveHandle();
    return KRET_OK;
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs) {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    (void)Recv(output_addr, output_size_list_[0] / sizeof(T), nccl_data_type_, src_rank_,
               reinterpret_cast<cudaStream_t>(cuda_stream_), group_name_);
    return true;
  }
  int src_rank_;
  bool is_null_input_{false};
  int unit_size_{0};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_
