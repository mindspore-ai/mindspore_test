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

#include "kernel/gpu/environ/environ_gpu_set.h"
#include "kernel/environ_manager.h"
#include "common/common_utils.h"
#include "plugin/res_manager/gpu/device/gpu_memory_allocator.h"

namespace mindspore {
namespace kernel {
int EnvironSetGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (!EnvironMgr::GetInstance().CheckEnvInput(primitive_, inputs, outputs)) {
    MS_LOG(ERROR) << "The input checks invalid, kernel: " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }

  // Check the output handle.
  auto handle_type = outputs[kIndex0]->dtype_id();
  const auto &handle_shapes = outputs[kIndex0]->GetShapeVector();
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(ERROR) << "The output handle checks invalid, kernel: " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }

  value_type_attr_ = TypeId(GetValue<int>(primitive_->GetAttr(kEnvValueTypeAttr)));
  MS_LOG(INFO) << "The EnvironSet kernel " << kernel_name_ << " value type: " << value_type_attr_;
  handle_size_ = sizeof(int64_t);
  key_size_ = sizeof(int64_t);

  auto value_type = inputs[kIndex2]->dtype_id();
  const auto &value_shapes = inputs[kIndex2]->GetShapeVector();
  value_size_ = GetTypeByte(TypeIdToType(value_type));
  for (auto &i : value_shapes) {
    value_size_ *= static_cast<size_t>(i);
  }

  output_size_list_.clear();
  output_size_list_.push_back(handle_size_);
  return KRET_OK;
}

bool EnvironSetGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto input_handle = GetDeviceAddress<int64_t>(inputs, kIndex0);
  auto input_key = GetDeviceAddress<int64_t>(inputs, kIndex1);
  auto input_value = GetDeviceAddress<void>(inputs, kIndex2);
  auto output_handle = GetDeviceAddress<int64_t>(outputs, kIndex0);

  // Get host handle and host key.
  int64_t host_handle = 0;
  int64_t host_key = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&host_handle, input_handle, handle_size_, cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Get handle failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&host_key, input_key, key_size_, cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Get key failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Sync stream failed.");

  // Alloc the value address, and free in the step end.
  auto value_ptr = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(value_size_);
  MS_EXCEPTION_IF_NULL(value_ptr);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(value_ptr, input_value, value_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Copy value failed.");

  // Set env member.
  const auto &env = EnvironMgr::GetInstance().Get(host_handle);
  if (env == nullptr) {
    MS_LOG(EXCEPTION) << "Get the env failed, handle: " << host_handle << ", key: " << host_key;
  }
  auto env_value = std::make_shared<EnvironValue>(value_ptr, value_size_, value_type_attr_, kGPUDevice);
  env->Set(host_key, env_value);

  // Copy output handle.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_handle, input_handle, handle_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "Copy output handle failed.");

  return true;
}
}  // namespace kernel
}  // namespace mindspore
