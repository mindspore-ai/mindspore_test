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

#include "kernel/cpu/native/sparse_tensor_to_csr_sparse_matrix_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace sparse_tensor_to_csr_sparse_matrix_cpu {
namespace {
constexpr int64_t kRankWithoutBatch = 2;
constexpr int64_t kRankWithBatch = 3;
constexpr int64_t kZero = 0;
constexpr int64_t kOne = 1;
constexpr int64_t kTwo = 2;
constexpr int64_t kSparseTensorToCSRSparseMatrixInputsNum = 3;
constexpr int64_t kSparseTensorToCSRSparseMatrixOutputsNum = 5;
constexpr int64_t kInitPrevBatch = -1;
constexpr char kKernelName[] = "SparseTensorToCSRSparseMatrix";

#define SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8) \
  KernelAttr()                                                                        \
    .AddInputAttr(kNumberType##t1)                                                    \
    .AddInputAttr(kNumberType##t2)                                                    \
    .AddInputAttr(kNumberType##t3)                                                    \
    .AddOutputAttr(kNumberType##t4)                                                   \
    .AddOutputAttr(kNumberType##t5)                                                   \
    .AddOutputAttr(kNumberType##t6)                                                   \
    .AddOutputAttr(kNumberType##t7)                                                   \
    .AddOutputAttr(kNumberType##t8)
}  // namespace

bool SparseTensorToCSRSparseMatrixCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  indice_type_ = inputs.at(kIndex0)->dtype_id();
  value_type_ = inputs.at(kIndex1)->dtype_id();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseTensorToCSRSparseMatrixInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseTensorToCSRSparseMatrixOutputsNum, kernel_name_);
  return true;
}

int SparseTensorToCSRSparseMatrixCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto x_indices_shape = inputs.at(kIndex0)->GetShapeVector();
  total_nnz_ = x_indices_shape.at(0);
  auto input_shape = inputs.at(kIndex2)->GetShapeVector();
  rank_ = input_shape.at(0);
  return KRET_OK;
}

bool SparseTensorToCSRSparseMatrixCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                                       const std::vector<kernel::KernelTensor *> &,
                                                       const std::vector<kernel::KernelTensor *> &outputs) {
  switch (indice_type_) {
    case kNumberTypeInt32:
      switch (value_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int32_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int32_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int32_t, complex64>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int32_t, complex128>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(value_type_)->ToString() << ".";
      }
      break;
    case kNumberTypeInt64:
      switch (value_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int64_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int64_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int64_t, complex64>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int64_t, complex128>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(value_type_)->ToString() << ".";
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of indices should be int32 or int64, "
                        << "but got " << TypeIdToType(indice_type_)->ToString() << ".";
  }
  return true;
}

template <typename indiceT, typename valueT>
void SparseTensorToCSRSparseMatrixCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                             const std::vector<kernel::KernelTensor *> &outputs) {
  const int64_t shift = (rank_ == kRankWithoutBatch) ? kZero : kOne;
  num_rows_ = GetDeviceAddress<indiceT>(inputs, kIndex2)[shift];
  indiceT *x_indices = GetDeviceAddress<indiceT>(inputs, kIndex0);
  valueT *x_values = GetDeviceAddress<valueT>(inputs, kIndex1);
  indiceT *x_dense_shape = GetDeviceAddress<indiceT>(inputs, kIndex2);
  batch_size_ = (rank_ == kRankWithoutBatch) ? kOne : x_dense_shape[kZero];
  indiceT *y_dense_shape_addr = GetDeviceAddress<indiceT>(outputs, kIndex0);
  indiceT *y_batch_pointers_addr = GetDeviceAddress<indiceT>(outputs, kIndex1);
  indiceT *y_row_pointers_addr = GetDeviceAddress<indiceT>(outputs, kIndex2);
  indiceT *y_col_indices_addr = GetDeviceAddress<indiceT>(outputs, kIndex3);
  valueT *y_values_addr = GetDeviceAddress<valueT>(outputs, kIndex4);

  for (int64_t i = kZero; i < rank_; i++) {
    y_dense_shape_addr[i] = x_dense_shape[i];
  }

  for (int64_t i = kZero; i < total_nnz_; i++) {
    y_values_addr[i] = x_values[i];
  }

  for (int64_t i = kZero; i < batch_size_ * (num_rows_ + 1); i++) {
    y_row_pointers_addr[i] = indiceT(kZero);
  }

  int64_t prev_batch = kInitPrevBatch;
  if (rank_ == kRankWithoutBatch) {
    y_batch_pointers_addr[kZero] = indiceT(kZero);
    ++prev_batch;
    for (int64_t i = kZero; i < total_nnz_; ++i) {
      y_row_pointers_addr[x_indices[i * rank_] + kOne] += indiceT(kOne);
      y_col_indices_addr[i] = x_indices[i * rank_ + kOne];
    }
  } else {
    for (int64_t i = kZero; i < total_nnz_; ++i) {
      int64_t cur_batch = static_cast<int64_t>(x_indices[i * rank_]);
      y_row_pointers_addr[cur_batch * (num_rows_ + kOne) + x_indices[i * rank_ + kOne] + kOne] += kOne;
      y_col_indices_addr[i] = x_indices[i * rank_ + kTwo];
      while (prev_batch < cur_batch) {
        y_batch_pointers_addr[prev_batch + kOne] = indiceT(i);
        ++prev_batch;
      }
    }
  }
  while (prev_batch < batch_size_) {
    y_batch_pointers_addr[prev_batch + kOne] = total_nnz_;
    ++prev_batch;
  }
  for (int64_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    auto *row_ptr_batch = y_row_pointers_addr + batch_idx * (num_rows_ + kOne);
    (void)std::partial_sum(row_ptr_batch, row_ptr_batch + num_rows_ + kOne, row_ptr_batch);
  }
}
std::vector<KernelAttr> SparseTensorToCSRSparseMatrixCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int32, Float32, Int32, Int32, Int32, Int32, Int32, Float32),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int32, Float64, Int32, Int32, Int32, Int32, Int32, Float64),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int32, Complex64, Int32, Int32, Int32, Int32, Int32, Complex64),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int32, Complex128, Int32, Int32, Int32, Int32, Int32, Complex128),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int64, Float32, Int64, Int64, Int64, Int64, Int64, Float32),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int64, Float64, Int64, Int64, Int64, Int64, Int64, Float64),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int64, Complex64, Int64, Int64, Int64, Int64, Int64, Complex64),
    SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_ADD_KERNEL(Int64, Complex128, Int64, Int64, Int64, Int64, Int64, Complex128)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixCpuKernelMod);
}  // namespace sparse_tensor_to_csr_sparse_matrix_cpu
}  // namespace kernel
}  // namespace mindspore
