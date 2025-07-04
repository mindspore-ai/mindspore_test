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

#include <stdio.h>
#include <math.h>
#include "kernel/gpu/cuda_impl/cuda_ops/sigmoid_impl.cuh"

template <typename T>
__global__ void SigmoidKernel(size_t size, const T *input, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(1) / (static_cast<T>(1) + exp(-input[pos]));
  }
}

template <typename T>
cudaError_t Sigmoid(size_t size, const T *input, T *output, cudaStream_t cuda_stream, const uint32_t device_id) {
  SigmoidKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Sigmoid<float>(size_t size, const float *input, float *output,
                                                    cudaStream_t cuda_stream, const uint32_t device_id);
