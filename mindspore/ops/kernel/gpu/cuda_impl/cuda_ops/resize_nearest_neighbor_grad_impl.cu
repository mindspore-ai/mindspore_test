/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/resize_nearest_neighbor_grad_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/upsample.cuh"

template <typename T, typename S>
__global__ void ResizeNearestNeighborGrad(const int input_size, const T *input, const int s1, const int s2,
                                          const int s3, const int s4, S *output, const int d1, const int d2,
                                          const int d3, const int d4, bool align_corners, float h_scale,
                                          float w_scale) {
  // initialization
  // HalfPixelCenters false
  int output_pos;
  int pos_array[RESIZENEARESTNEIGHBORGRAD_DIMENSION];
  int out_height = d3;
  int out_width = d4;
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
  int in_h;
  int in_w;

  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < (input_size); pos += gridDim.x * blockDim.x) {
    pos_array[0] = pos / (s2 * s3 * s4) % s1;
    pos_array[1] = pos / (s3 * s4) % s2;
    pos_array[2] = pos / (s4) % s3;
    pos_array[3] = pos % s4;
    in_h = pos_array[2];
    in_w = pos_array[3];
    const int out_y =
      min((align_corners) ? static_cast<int>(roundf(in_h * h_scale)) : static_cast<int>(floorf(in_h * h_scale)),
          out_height - 1);
    const int out_x =
      min((align_corners) ? static_cast<int>(roundf(in_w * w_scale)) : static_cast<int>(floorf(in_w * w_scale)),
          out_width - 1);
    // pos_array[0] N, pos_array[1] C, out_y H, out_x W
    output_pos = pos_array[0] * d2 * d3 * d4 + pos_array[1] * d3 * d4 + out_y * d4 + out_x;
    MsAtomicAdd(&output[output_pos], static_cast<S>(input[pos]));
  }
}

template <typename T>
cudaError_t CalResizeNearestNeighborGrad(float *work, const int input_size, const T *input, const int s1, const int s2,
                                         const int s3, const int s4, T *output, const int d1, const int d2,
                                         const int d3, const int d4, bool align_corners, float h_scale, float w_scale,
                                         cudaStream_t cuda_stream) {
  ResizeNearestNeighborGrad<T, T><<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
    input_size, input, s1, s2, s3, s4, output, d1, d2, d3, d4, align_corners, h_scale, w_scale);
  return GetCudaStatus();
}

template <>
cudaError_t CalResizeNearestNeighborGrad(float *work, const int input_size, const half *input, const int s1,
                                         const int s2, const int s3, const int s4, half *output, const int d1,
                                         const int d2, const int d3, const int d4, bool align_corners, float h_scale,
                                         float w_scale, cudaStream_t cuda_stream) {
  ResizeNearestNeighborGrad<half, float><<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
    input_size, input, s1, s2, s3, s4, work, d1, d2, d3, d4, align_corners, h_scale, w_scale);
  const int num_kernels = d1 * d2 * d3 * d4;
  CudaMemcpyDeviceToDevice<float, half>
    <<<GET_BLOCKS(num_kernels), GET_THREADS, 0, cuda_stream>>>(num_kernels, work, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalResizeNearestNeighborGrad<double>(
  float *work, const int input_size, const double *input, const int s1, const int s2, const int s3, const int s4,
  double *output, const int d1, const int d2, const int d3, const int d4, bool align_corners, float h_scale,
  float w_scale, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeNearestNeighborGrad<float>(
  float *work, const int input_size, const float *input, const int s1, const int s2, const int s3, const int s4,
  float *output, const int d1, const int d2, const int d3, const int d4, bool align_corners, float h_scale,
  float w_scale, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeNearestNeighborGrad<half>(float *work, const int input_size,
                                                                        const half *input, const int s1, const int s2,
                                                                        const int s3, const int s4, half *output,
                                                                        const int d1, const int d2, const int d3,
                                                                        const int d4, bool align_corners, float h_scale,
                                                                        float w_scale, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeNearestNeighborGrad<int>(float *work, const int input_size,
                                                                       const int *input, const int s1, const int s2,
                                                                       const int s3, const int s4, int *output,
                                                                       const int d1, const int d2, const int d3,
                                                                       const int d4, bool align_corners, float h_scale,
                                                                       float w_scale, cudaStream_t cuda_stream);
