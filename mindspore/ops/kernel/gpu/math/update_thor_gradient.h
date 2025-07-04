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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UPDATE_THOR_GRADIENT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UPDATE_THOR_GRADIENT_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/cuda_ops/convert_gradient_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace kernel {
struct GradientSize {
  size_t batch_h;
  size_t batch_w;
  size_t h;
  size_t w;
  size_t ori_h;
  size_t ori_w;
  size_t pad_h;
  size_t pad_w;
  bool need_convert;
  cudaDataType_t dtype;
};
template <typename T>
class UpdateThorGradientGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpdateThorGradientGpuKernelMod() : split_dim(128), handle_(nullptr) {}
  ~UpdateThorGradientGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    auto input1_addr = GetDeviceAddress<T>(inputs, 0);
    auto input2_addr = GetDeviceAddress<T>(inputs, 1);
    auto input3_addr = GetDeviceAddress<T>(inputs, 2);
    auto workspace1_addr = GetDeviceAddress<T>(workspace, 0);
    T *workspace2_addr = nullptr;
    T *workspace3_addr = nullptr;
    if (gradient_size.need_convert) {
      workspace2_addr = GetDeviceAddress<T>(workspace, 1);
      workspace3_addr = GetDeviceAddress<T>(workspace, 2);
    }
    T *workspace4_addr = nullptr;
    cudaError_t status = cudaErrorNotReady;
    auto output_addr = GetDeviceAddress<T>(outputs, 0);
    if (gradient_size.pad_h != 0 || gradient_size.pad_w != 0) {
      workspace4_addr = GetDeviceAddress<T>(workspace, 3);
      const size_t size = (gradient_size.ori_h + gradient_size.pad_h) * (gradient_size.ori_w + gradient_size.pad_w);
      status = CalPad(size, input2_addr, 1, 1, gradient_size.ori_h, gradient_size.ori_w,
                      gradient_size.ori_h + gradient_size.pad_h, gradient_size.ori_w + gradient_size.pad_w, 0, 0, 0.0,
                      workspace4_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      cudaMemsetAsync(workspace1_addr, 0,
                      gradient_size.w * gradient_size.h * gradient_size.batch_w * gradient_size.batch_h * sizeof(T),
                      reinterpret_cast<cudaStream_t>(stream_ptr));
      input2_addr = workspace4_addr;
    }
    const float alpha = 1;
    const float beta = 0;
    const int lda = SizeToInt(gradient_size.h);
    const int ldb = SizeToInt(gradient_size.ori_w + gradient_size.pad_w);
    const int ldc = SizeToInt(gradient_size.ori_w + gradient_size.pad_w);

    auto stride_a = SizeToInt(gradient_size.h * gradient_size.h);
    auto stride_b = SizeToInt(gradient_size.h * (gradient_size.ori_w + gradient_size.pad_w));
    auto stride_c = SizeToInt(gradient_size.h * (gradient_size.ori_w + gradient_size.pad_w));

    try {
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, SizeToInt(gradient_size.ori_w),
                                   SizeToInt(gradient_size.h), SizeToInt(gradient_size.h), &alpha, input2_addr,
                                   gradient_size.dtype, ldb, stride_b, input1_addr, gradient_size.dtype, lda, stride_a,
                                   &beta, workspace1_addr, gradient_size.dtype, ldc, stride_c, gradient_size.batch_h,
                                   CUDA_R_32F, algo_),
        "cublasSgemm Call Fail");
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", encountered an exception: " << e.what()
                        << " when invoke cubals cublasGemmStridedBatchedEx";
    }

    auto r_input_addr = workspace1_addr;
    if (gradient_size.need_convert) {
      size_t size = gradient_size.batch_w * gradient_size.batch_h * gradient_size.w * gradient_size.h;
      status = ConvertGradient(size, gradient_size.h, gradient_size.w, gradient_size.batch_w,
                               gradient_size.batch_w * gradient_size.w, workspace1_addr, workspace2_addr,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      r_input_addr = workspace2_addr;
    }

    const int lda_r = SizeToInt(gradient_size.w);
    const int ldb_r = SizeToInt(gradient_size.w);
    const int ldc_r = SizeToInt(gradient_size.w);

    stride_a = SizeToInt(gradient_size.h * gradient_size.w);
    stride_b = SizeToInt(gradient_size.w * gradient_size.w);
    stride_c = SizeToInt(gradient_size.h * gradient_size.w);
    auto r_output_addr = output_addr;
    if (gradient_size.need_convert) {
      r_output_addr = workspace3_addr;
    }
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, SizeToInt(gradient_size.w),
                                 SizeToInt(gradient_size.h), SizeToInt(gradient_size.w), &alpha, input3_addr,
                                 gradient_size.dtype, ldb_r, stride_b, r_input_addr, gradient_size.dtype, lda_r,
                                 stride_a, &beta, r_output_addr, gradient_size.dtype, ldc_r, stride_c,
                                 gradient_size.batch_h * gradient_size.batch_w, CUDA_R_32F, algo_),
      "cublasSgemm Call Fail");
    if (gradient_size.need_convert) {
      size_t size = gradient_size.batch_w * gradient_size.batch_h * gradient_size.w * gradient_size.h;
      if (gradient_size.pad_h == 0 && gradient_size.pad_w == 0) {
        status = ConvertGradientBack(size, gradient_size.h, gradient_size.w, gradient_size.batch_w,
                                     gradient_size.batch_w * gradient_size.w, r_output_addr, output_addr,
                                     reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        status = ConvertGradientBack(size, gradient_size.h, gradient_size.w, gradient_size.ori_h, gradient_size.ori_w,
                                     gradient_size.batch_w, gradient_size.ori_w, r_output_addr, output_addr,
                                     reinterpret_cast<cudaStream_t>(stream_ptr));
      }
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    output_size_list_.clear();
    (void)SetProperty(primitive_, inputs, outputs);
    InitSizeLists();
    return KRET_OK;
  }

 protected:
  void InitSizeLists() {
    size_t unit_size = sizeof(T);

    size_t output_size = gradient_size.ori_h * gradient_size.ori_w * unit_size;
    output_size_list_.push_back(output_size);

    size_t workspace_size_ =
      gradient_size.w * gradient_size.h * gradient_size.batch_w * gradient_size.batch_h * unit_size;
    workspace_size_list_.push_back(workspace_size_);

    if (gradient_size.need_convert) {
      workspace_size_ = gradient_size.w * gradient_size.h * gradient_size.batch_w * gradient_size.batch_h * unit_size;
      workspace_size_list_.push_back(workspace_size_);
      workspace_size_ = gradient_size.w * gradient_size.h * gradient_size.batch_w * gradient_size.batch_h * unit_size;
      workspace_size_list_.push_back(workspace_size_);
    }

    if (gradient_size.pad_h != 0 || gradient_size.pad_w != 0) {
      workspace_size_ =
        (gradient_size.ori_w + gradient_size.pad_w) * (gradient_size.ori_h + gradient_size.pad_h) * unit_size;
      workspace_size_list_.push_back(workspace_size_);
    }
  }

 private:
  void SetProperty(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                   const std::vector<KernelTensor *> &outputs) {
    const auto &matrix_a_shape = inputs[kIndex0]->GetShapeVector();
    const auto &shape_signed = inputs[kIndex1]->GetShapeVector();
    auto gradient_shape = Convert2SizeTClipNeg(shape_signed);
    const auto &matrix_g_shape = inputs[kIndex2]->GetShapeVector();
    if (AnfAlgo::IsShapesDynamic({matrix_a_shape, shape_signed, matrix_g_shape})) {
      return;
    }
    is_null_input_ = CHECK_SHAPE_NULL(matrix_a_shape, kernel_name_, "matrix_a") ||
                     CHECK_SHAPE_NULL(gradient_shape, kernel_name_, "gradient") ||
                     CHECK_SHAPE_NULL(matrix_g_shape, kernel_name_, "matrix_g");
    if (is_null_input_) {
      return;
    }

    split_dim = LongToSize(GetValue<int64_t>(primitive->GetAttr("split_dim")));
    if (split_dim == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", divide by zero, split_dim cannot be 0, but got " << split_dim;
    }
    gradient_size.batch_h = gradient_shape[0] / split_dim;
    gradient_size.batch_w = gradient_shape[1] / split_dim;
    if (gradient_size.batch_h * split_dim != gradient_shape[0]) {
      gradient_size.batch_h += 1;
      if (gradient_shape[0] > split_dim) {
        gradient_size.h = split_dim;
        gradient_size.pad_h = gradient_size.batch_h * split_dim - gradient_shape[0];
      } else {
        gradient_size.h = gradient_shape[0];
        gradient_size.pad_h = 0;
      }
    } else {
      gradient_size.h = split_dim;
      gradient_size.pad_h = 0;
    }

    if (gradient_size.batch_w * split_dim != gradient_shape[1]) {
      gradient_size.batch_w += 1;
      if (gradient_shape[1] > split_dim) {
        gradient_size.w = split_dim;
        gradient_size.pad_w = gradient_size.batch_w * split_dim - gradient_shape[1];
      } else {
        gradient_size.w = gradient_shape[1];
        gradient_size.pad_w = 0;
      }
    } else {
      gradient_size.w = split_dim;
      gradient_size.pad_w = 0;
    }

    if (gradient_size.batch_w * gradient_size.w <= split_dim) {
      gradient_size.need_convert = false;
    } else {
      gradient_size.need_convert = true;
    }

    gradient_size.ori_w = gradient_shape[1];
    gradient_size.ori_h = gradient_shape[0];
    gradient_size.dtype = GetCudaDataType(TypeIdLabel(inputs[kIndex1]->dtype_id()));
  }

  size_t split_dim;
  bool is_null_input_;
  struct GradientSize gradient_size;
  cublasHandle_t handle_;
  cublasGemmAlgo_t algo_ = CUBLAS_GEMM_DEFAULT;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UPDATE_THOR_GRADIENT_GPU_KERNEL_H_
