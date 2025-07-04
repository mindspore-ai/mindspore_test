/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <cuda_runtime.h>
#include "kernel/gpu/cuda_impl/cuda_ops/mish_impl.cuh"
#include "include/cuda_fp16.h"
template <typename T>

__global__ void MishKernel(const size_t size, const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = input_addr[pos] * tanh(logf(1. + expf(input_addr[pos])));
  }
}

template <>
__global__ void MishKernel(const size_t size, const half *input_addr, half *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = __half2float(input_addr[pos]) * tanh(logf(1. + exp(__half2float(input_addr[pos]))));
  }
}

template <>
__global__ void MishKernel(const size_t size, const double *input_addr, double *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = input_addr[pos] * tanh(logf(1. + exp(input_addr[pos])));
  }
}

template <typename T>
cudaError_t Mish(const size_t size, const T *input_addr, T *output_addr, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  MishKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_addr, output_addr);
  return GetCudaStatus();
}

template <>
cudaError_t Mish(const size_t size, const half *input_addr, half *output_addr, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  MishKernel<half>
    <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_addr, output_addr);
  return GetCudaStatus();
}

template <>
cudaError_t Mish(const size_t size, const double *input_addr, double *output_addr, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  MishKernel<double>
    <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_addr, output_addr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Mish<float>(const size_t size, const float *input_addr, float *output_addr,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Mish<half>(const size_t size, const half *input_addr, half *output_addr,
                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Mish<double>(const size_t size, const double *input_addr, double *output_addr,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
