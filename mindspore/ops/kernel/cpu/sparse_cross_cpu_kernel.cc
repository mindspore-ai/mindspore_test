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

#include "kernel/cpu/sparse_cross_cpu_kernel.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <vector>
#include "mindspore/ops/op_def/math_ops.h"

namespace mindspore {
namespace kernel {
namespace sparse_cross_cpu {
namespace {
constexpr int64_t kInputsNum = 4;
constexpr int64_t kOutputsNum = 3;
constexpr int64_t kInputIndices = 0;
constexpr int64_t kInputValue = 1;
constexpr int64_t kInputShape = 2;
constexpr int64_t kInputDense = 3;
constexpr int64_t kOutputIndices = 0;
constexpr int64_t kOutputValue = 1;
constexpr int64_t kOutputShape = 2;
}  // namespace

template <typename T>
class TensorColumnBase {
 public:
  virtual int64_t FeatureCount(int64_t batch) const = 0;

  virtual T Feature(int64_t batch, int64_t n) const = 0;

  virtual ~TensorColumnBase() {}
};

template <typename T>
class SparseTensorColumn : public TensorColumnBase<T> {
 public:
  SparseTensorColumn(const T *values_ptr, const std::vector<int64_t> &feature_counts,
                     const std::vector<int64_t> &feature_start_indices)
      : values_ptr_(values_ptr),
        feature_counts_(std::make_unique<std::vector<int64_t>>(feature_counts)),
        feature_start_indices_(std::make_unique<std::vector<int64_t>>(feature_start_indices)) {
    if (feature_counts_->size() != feature_start_indices_->size()) {
      MS_LOG(EXCEPTION) << "For SparseTensor, feature_counts_ is not equal to feature_start_indices_.";
    }
  }

  int64_t FeatureCount(int64_t batch) const override { return (*feature_counts_)[batch]; }

  T Feature(int64_t batch, int64_t n) const override;

  ~SparseTensorColumn() override = default;

 private:
  const T *values_ptr_;
  std::unique_ptr<std::vector<int64_t>> feature_counts_;
  std::unique_ptr<std::vector<int64_t>> feature_start_indices_;
};

template <>
int64_t SparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n) const {
  const int64_t start = (*feature_start_indices_)[batch];
  return static_cast<int64_t>(values_ptr_[start + n]);
}

template <typename T>
class DenseTensorColumn : public TensorColumnBase<T> {
 public:
  DenseTensorColumn(const T *base_ptr, int64_t row_stride) : base_ptr_(base_ptr), row_stride_(row_stride) {}
  int64_t FeatureCount(int64_t) const override { return row_stride_; }
  T Feature(int64_t batch, int64_t n) const override;
  ~DenseTensorColumn() override {}

 private:
  const T *base_ptr_;
  int64_t row_stride_;
};

template <>
int64_t DenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n) const {
  return base_ptr_[batch * row_stride_ + n];
}

template <typename OutType>
class OutputUpdaterDirect {
 public:
  OutputUpdaterDirect(const std::vector<int64_t> &output_start_indices, int64_t *indices_out, int64_t *values_out)
      : output_start_indices_(output_start_indices), indices_out_(indices_out), values_out_(values_out) {}

  void Update(int64_t batch_index, int64_t cross_count, OutType cross) {
    const int64_t output_index = output_start_indices_[batch_index] + cross_count;
    const int64_t idx_offset = output_index * 2;
    indices_out_[idx_offset] = batch_index;
    indices_out_[idx_offset + 1] = cross_count;
    values_out_[output_index] = cross;
  }

 private:
  const std::vector<int64_t> &output_start_indices_;
  int64_t *indices_out_;
  int64_t *values_out_;
};

class HashCrosser {
 public:
  explicit HashCrosser(const std::vector<std::unique_ptr<TensorColumnBase<int64_t>>> &columns,
                       const int64_t num_buckets, const uint64_t hash_key)
      : columns_(columns), num_buckets_(num_buckets), hash_key_(hash_key) {}

  uint64_t ShiftMix(const uint64_t val) const { return val ^ (val >> 47); }
  uint64_t FingerprintCat64(uint64_t fp1, const uint64_t fp2) const {
    static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
    uint64_t result = fp1 ^ kMul;
    result ^= ShiftMix(fp2 * kMul) * kMul;
    result *= kMul;
    result = ShiftMix(result) * kMul;
    result = ShiftMix(result);
    return result;
  }

  int64_t Generate(const int64_t batch_index, const std::vector<int64_t> &permutation) const {
    uint64_t hashed_output = hash_key_;
    for (size_t i = 0; i < permutation.size(); ++i) {
      uint64_t hash_i = static_cast<uint64_t>(columns_[i]->Feature(batch_index, permutation[i]));
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      return hashed_output % static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    }
  }

 private:
  const std::vector<std::unique_ptr<TensorColumnBase<int64_t>>> &columns_;
  const int64_t num_buckets_;
  const uint64_t hash_key_;
};

template <typename T>
class ProductIterator {
 public:
  explicit ProductIterator(const std::vector<std::unique_ptr<TensorColumnBase<T>>> &columns, int64_t batch_index)
      : columns_(columns), batch_index_(batch_index) {
    next_permutation_.resize(columns_.size(), 0);
    has_next_ = true;
    for (uint32_t i = 0; i < columns_.size(); i++) {
      if (columns_[i]->FeatureCount(batch_index_) == 0) {
        has_next_ = false;
        break;
      }
    }
  }

  const std::vector<int64_t> &Current() const { return next_permutation_; }

  bool Advance() {
    bool carry = true;
    for (int64_t i = static_cast<int64_t>(next_permutation_.size() - 1); i >= 0; i--) {
      if (carry) {
        next_permutation_[i] = next_permutation_[i] + 1;
      }
      if (next_permutation_[i] == columns_[i]->FeatureCount(batch_index_)) {
        next_permutation_[i] = 0;
      } else {
        carry = false;
        break;
      }
    }
    has_next_ = !carry;
    return has_next_;
  }

  bool HasNext() const { return has_next_; }

 private:
  bool has_next_;
  const std::vector<std::unique_ptr<TensorColumnBase<T>>> &columns_;
  const int64_t batch_index_;
  std::vector<int64_t> next_permutation_;
};

template <bool HASHED_OUTPUT, typename T>
struct CrossTraits;
template <>
struct CrossTraits<true, int64_t> {
  using Crosser = HashCrosser;
  using Updater = OutputUpdaterDirect<int64_t>;
};

bool SparseCrossCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  hash_key_ = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("hash_key")));
  hash_out_ = GetValue<bool>(primitive_->GetAttr("hashed_output"));
  num_buckets_ = GetValue<int64_t>(primitive_->GetAttr("num_buckets"));
  return true;
}

int SparseCrossCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK && ret != KRET_UNKNOWN_OUT_SHAPE) {
    return ret;
  }

  N_ = GetValue<int64_t>(primitive_->GetAttr("N"));
  return KRET_OK;
}

void SparseCrossCpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  int64_t kSparseTensorRank = 2;
  outputs[kIndex0]->SetShapeVector(ShapeVector({indices_row_, kSparseTensorRank}));
  outputs[kIndex1]->SetShapeVector(ShapeVector({indices_row_}));
  outputs[kIndex2]->SetShapeVector(ShapeVector({kSparseTensorRank}));
  outputs[kIndex0]->set_size(LongToSize(indices_row_ * kSparseTensorRank) *
                             UnitSizeInBytes(outputs[kIndex0]->dtype_id()));
  outputs[kIndex1]->set_size(LongToSize(indices_row_) * UnitSizeInBytes(outputs[kIndex1]->dtype_id()));
  outputs[kIndex2]->set_size(LongToSize(kSparseTensorRank) * UnitSizeInBytes(outputs[kIndex2]->dtype_id()));
}

void ExtractFeatureData(const std::vector<std::vector<int64_t>> &indices_list_in, int64_t batch_size,
                        std::vector<std::vector<int64_t>> *const feature_counts,
                        std::vector<std::vector<int64_t>> *const feature_start_indices) {
  std::vector<int64_t> current_row(indices_list_in.size(), 0);
  const uint32_t stride = 2;
  for (int64_t b = 0; b < batch_size; b++) {
    for (uint32_t i = 0; i < indices_list_in.size(); i++) {
      int64_t feature_count = 0;
      const int64_t start_index = current_row[i];
      const int64_t entries = static_cast<int64_t>(indices_list_in[i].size() / stride);
      while (current_row[i] < entries && indices_list_in[i][current_row[i] * stride] == b) {
        feature_count++;
        current_row[i]++;
      }
      (*feature_counts)[i].push_back(feature_count);
      (*feature_start_indices)[i].push_back(start_index);
    }
  }
}

template <typename T>
int64_t CrossCountByBatchIndex(const std::vector<std::unique_ptr<TensorColumnBase<T>>> &columns, int64_t batch_index) {
  int64_t cross_count = 1;
  for (size_t i = 0; i < columns.size(); i++) {
    const auto feature_count = columns[i]->FeatureCount(batch_index);
    if (feature_count == 0) {
      return 0;
    }
    cross_count *= feature_count;
  }
  return cross_count;
}

template <typename T>
std::vector<std::unique_ptr<TensorColumnBase<T>>> GenerateColumnsFromInput(
  const std::vector<std::vector<int64_t>> &indices_list_in, const std::vector<std::vector<T>> &values_list_in,
  const std::vector<std::vector<int64_t>> &shapes_list_in, const std::vector<const T *> &dense_base_ptrs,
  const std::vector<int64_t> &dense_row_strides, int64_t batch_size) {
  std::vector<std::unique_ptr<TensorColumnBase<T>>> columns;
  const int64_t number_of_columns = static_cast<int64_t>(shapes_list_in.size());
  std::vector<std::vector<int64_t>> feature_counts(number_of_columns, std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(number_of_columns, std::vector<int64_t>());
  ExtractFeatureData(indices_list_in, batch_size, &feature_counts, &feature_start_indices);
  columns.reserve(values_list_in.size() + dense_base_ptrs.size());
  for (uint32_t i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(std::unique_ptr<TensorColumnBase<T>>(
      new SparseTensorColumn<T>(values_list_in[i].data(), feature_counts[i], feature_start_indices[i])));
  }
  for (uint32_t i = 0; i < dense_base_ptrs.size(); ++i) {
    columns.emplace_back(
      std::unique_ptr<TensorColumnBase<T>>(new DenseTensorColumn<T>(dense_base_ptrs[i], dense_row_strides[i])));
  }
  return columns;
}

template <typename T>
void CreateOutputTensors(const std::vector<std::unique_ptr<TensorColumnBase<T>>> &columns, int64_t batch_size,
                         std::vector<int64_t> *output_start_indices, int64_t *out_num, int64_t *shape_vec) {
  int64_t cross_count_total = 0;
  int64_t max_cross_count = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    (*output_start_indices)[b] = cross_count_total;
    const auto cross_count = CrossCountByBatchIndex<T>(columns, b);
    max_cross_count = std::max(max_cross_count, cross_count);
    cross_count_total += cross_count;
  }
  shape_vec[0] = batch_size;
  shape_vec[1] = max_cross_count;
  *out_num = cross_count_total;
}

template <bool HASHED_OUTPUT, typename T>
bool SparseCrossCpuKernelMod::SparseCrossCann(const std::vector<std::unique_ptr<TensorColumnBase<T>>> &columns,
                                              int64_t batch_size, const std::vector<int64_t> &output_start_indices,
                                              const std::vector<kernel::KernelTensor *> &outputs) const {
  auto indices_out = GetDeviceAddress<int64_t>(outputs, kOutputIndices);
  auto values_out = GetDeviceAddress<int64_t>(outputs, kOutputValue);

  typename CrossTraits<HASHED_OUTPUT, T>::Crosser crosser(columns, num_buckets_, hash_key_);
  typename CrossTraits<HASHED_OUTPUT, T>::Updater updater(output_start_indices, indices_out, values_out);

  for (int64_t b = 0; b < batch_size; b++) {
    ProductIterator<T> product_iterator(columns, b);
    int64_t cross_count = 0;
    while (product_iterator.HasNext()) {
      const auto &permutation = product_iterator.Current();
      updater.Update(b, cross_count, crosser.Generate(b, permutation));
      cross_count++;
      if (!product_iterator.Advance()) break;
    }
  }
  return true;
}

int64_t fill(const std::vector<std::vector<int64_t>> &indices_list_in, const std::vector<std::vector<int64_t>> &,
             const std::vector<std::vector<int64_t>> &shapes_list_in,
             const std::vector<std::vector<int64_t>> &denses_list_in, const std::vector<kernel::KernelTensor *> &inputs,
             uint32_t sizen) {
  auto n_row = shapes_list_in[0][0];
  int64_t in_num = static_cast<int64_t>(sizen);
  std::vector<std::vector<int64_t>> rowno(in_num, std::vector<int64_t>(n_row, 0));
  uint32_t g_value = 2;
  for (uint32_t i = 0; i < sizen; i++) {
    for (uint32_t k = 0; k < indices_list_in[i].size(); k = k + g_value) {
      int64_t row = indices_list_in[i][k];
      rowno[i][row]++;
    }
  }
  uint32_t group_v = 3;
  std::vector<int64_t> dens(inputs.size() - sizen * group_v);
  for (uint32_t di = 0; di < inputs.size() - sizen * group_v; di++) {
    dens[di] = static_cast<int64_t>(denses_list_in[di].size()) / n_row;
  }
  int64_t indices_s = 0;
  for (int64_t m = 0; m < n_row; m++) {
    int64_t tmp = 1;
    for (uint32_t n = 0; n < sizen; n++) {
      tmp = tmp * rowno[n][m];
    }
    for (uint32_t di = 0; di < inputs.size() - sizen * group_v; di++) {
      tmp = tmp * dens[di];
    }
    indices_s = indices_s + tmp;
  }
  return indices_s;
}

template <typename T, typename S>
bool SparseCrossCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  uint32_t sizen = static_cast<uint32_t>(N_);
  size_t shape_size = inputs[kInputShape * sizen]->size() / sizeof(int64_t);
  for (unsigned int i = 0; i < sizen; i++) {
    if (shape_size != inputs[kInputShape * sizen + i]->size() / sizeof(int64_t)) {
      MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", the input COO sparse tensor shape dims is "
                        << inputs[kInputShape * sizen + i]->size() / sizeof(int64_t)
                        << ", not equal with the first COO sparse tensor dims : " << shape_size << ".";
    }
  }
  std::vector<std::vector<int64_t>> indices_list_in(sizen);
  for (uint32_t i = 0; i < sizen; ++i) {
    auto input1_ptr = static_cast<int64_t *>(inputs[kInputIndices + i]->device_ptr());
    uint32_t inputs_1 = inputs[kInputIndices + i]->size() / sizeof(int64_t);
    indices_list_in[i].reserve(inputs_1);
    for (uint32_t j = 0; j < inputs_1; j++) {
      indices_list_in[i].push_back(*(input1_ptr + j));
    }
  }
  std::vector<std::vector<int64_t>> values_list_in(sizen);
  for (uint32_t i = 0; i < sizen; ++i) {
    auto input1_ptr = static_cast<int64_t *>(inputs[kInputValue * sizen + i]->device_ptr());
    uint32_t inputs_1 = inputs[kInputValue * sizen + i]->size() / sizeof(int64_t);
    values_list_in[i].reserve(inputs_1);
    for (uint32_t j = 0; j < inputs_1; j++) {
      values_list_in[i].push_back(*(input1_ptr + j));
    }
  }
  std::vector<std::vector<int64_t>> shapes_list_in(sizen);
  for (uint32_t i = 0; i < sizen; ++i) {
    auto input1_ptr = static_cast<int64_t *>(inputs[kInputShape * sizen + i]->device_ptr());
    uint32_t inputs_1 = inputs[kInputShape * sizen + i]->size() / sizeof(int64_t);
    shapes_list_in[i].reserve(inputs_1);
    for (uint32_t j = 0; j < inputs_1; j++) {
      shapes_list_in[i].push_back(*(input1_ptr + j));
    }
  }
  const uint32_t d_n = inputs.size() - sizen * kIndex3;
  std::vector<std::vector<int64_t>> denses_list_in(d_n);
  std::vector<const int64_t *> dense_base_ptrs;
  std::vector<int64_t> dense_row_strides;
  dense_base_ptrs.reserve(d_n);
  dense_row_strides.reserve(d_n);
  for (uint32_t i = 0; i < d_n; ++i) {
    auto input2_ptr = static_cast<int64_t *>(inputs[kInputDense * sizen + i]->device_ptr());
    uint32_t inputs_2 = inputs[kInputDense * sizen + i]->size() / sizeof(int64_t);
    denses_list_in[i].reserve(inputs_2);
    for (uint32_t j = 0; j < inputs_2; j++) {
      denses_list_in[i].push_back(input2_ptr[j]);
    }
  }
  // Compute indices_row_ for shape update path
  indices_row_ = fill(indices_list_in, values_list_in, shapes_list_in, denses_list_in, inputs, sizen);
  if (!hash_out_) {
    MS_EXCEPTION(TypeError) << "For Op " << kernel_name_ << ", only support int64, so hashed_output should be true.";
  }

  // Determine batch size: prefer sparse shapes; fallback to dense-only behavior of original implementation
  int64_t batch_size = 0;
  if (!shapes_list_in.empty()) {
    batch_size = shapes_list_in[0][0];
  } else if (!denses_list_in.empty()) {
    batch_size = static_cast<int64_t>(denses_list_in[0].size());
  }
  // Set up base pointers and strides now that batch_size is known
  for (uint32_t i = 0; i < d_n; ++i) {
    const auto total = static_cast<int64_t>(denses_list_in[i].size());
    const int64_t row_stride = (batch_size > 0) ? (total / batch_size) : 0;
    dense_base_ptrs.push_back(denses_list_in[i].data());
    dense_row_strides.push_back(row_stride);
  }

  // Build columns and compute output offsets and shape
  std::vector<std::unique_ptr<TensorColumnBase<int64_t>>> columns = GenerateColumnsFromInput<int64_t>(
    indices_list_in, values_list_in, shapes_list_in, dense_base_ptrs, dense_row_strides, batch_size);

  auto out_shape = GetDeviceAddress<int64_t>(outputs, kOutputShape);
  std::vector<int64_t> output_start_indices(batch_size);
  int64_t out_num = 0;
  CreateOutputTensors(columns, batch_size, &output_start_indices, &out_num, out_shape);

  // Run cross and write directly to outputs
  bool res = SparseCrossCann<true, int64_t>(columns, batch_size, output_start_indices, outputs);
  if (!res) return false;
  return true;
}

const std::vector<std::pair<KernelAttr, SparseCrossCpuKernelMod::KernelRunFunc>> &SparseCrossCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SparseCrossCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddSkipCheckAttr(true), &SparseCrossCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseCross, SparseCrossCpuKernelMod);
}  // namespace sparse_cross_cpu
}  // namespace kernel
}  // namespace mindspore
