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

#include "kernel/gpu/rl/tensor_array_close_kernel.h"
#include "common/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::gpu::GPUTensorArray;
using mindspore::device::gpu::GPUTensorArrayPtr;
TensorArrayCloseKernelMod::TensorArrayCloseKernelMod() {}

bool TensorArrayCloseKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  return true;
}

int TensorArrayCloseKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorArrayCloseKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &, void *stream) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_ERROR_IF_NULL(handle_addr);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  MS_ERROR_IF_NULL(cuda_stream);
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'TensorArrayClose', get handle to host failed");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "cuda Stream Sync Failed");
  }
  GPUTensorArrayPtr tensors_ =
    std::dynamic_pointer_cast<GPUTensorArray>(TensorArrayMgr::GetInstance().GetTensorArray(handle));
  MS_ERROR_IF_NULL(tensors_);
  // Free device mem
  tensors_->Free();
  // Erase tensorarray
  if (!TensorArrayMgr::GetInstance().EraseTensorArray(handle)) {
    MS_LOG(EXCEPTION) << "Free tensorarray failed";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
