/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSECROSS_H_
#define AICPU_KERNELS_NORMALIZED_SPARSECROSS_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include "inc/ms_cpu_kernel.h"
#include "inc/ms_cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
class SparseCrossCpuKernel : public CpuKernel {
 public:
  SparseCrossCpuKernel() = default;
  ~SparseCrossCpuKernel() = default;

 protected:
  // template <bool HASHED_OUTPUT, typename InternalType>
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <bool HASHED_OUTPUT, typename InternalType>
  uint32_t SparseCrossCompute(CpuKernelContext &ctx);

  int64_t num_buckets_;
  uint64_t hash_key_;
};

template <typename ListType, typename ElementType>
class OpArgIterator {
 public:
  typedef ElementType *pointer;
  typedef const ElementType *const_pointer;
  typedef ElementType &reference;
  typedef const ElementType &const_reference;

  OpArgIterator(const ListType *list, int i) : list_(list), idx_(i) {}

  bool operator==(const OpArgIterator &rhs) {
    if (list_ == rhs.list_) {
      return idx_ == rhs.idx_;
    }
    return false;
  }

  bool operator!=(const OpArgIterator &rhs) {
    if (list_ == rhs.list_) {
      return idx_ != rhs.idx_;
    }
    return true;
  }

  OpArgIterator operator++() {  // prefix ++it
    ++idx_;
    return *this;
  }

  reference operator*() { return (*list_)[idx_]; }

  OpArgIterator operator++(int) {  // postfix it++
    OpArgIterator old_value = *this;
    ++idx_;
    return old_value;
  }

  pointer operator->() { return &(*list_)[idx_]; }

  const_reference operator*() const { return (*list_)[idx_]; }
  const_pointer operator->() const { return &(*list_)[idx_]; }

 private:
  const ListType *const list_;
  int idx_;
};

class OpInputList {
 public:
  using Iterator = OpArgIterator<OpInputList, const Tensor>;
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpInputList(CpuKernelContext *ctx, uint32_t start, uint32_t stop) : ctx_(ctx), start_(start), stop_(stop) {}
  OpInputList &operator=(const OpInputList &other) = default;
  OpInputList(const OpInputList &other) = default;
  Tensor *operator[](uint32_t i) const { return ctx_->Input(start_ + i); }
  uint32_t size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  CpuKernelContext *ctx_;  // not owned
  uint32_t start_;
  uint32_t stop_;
};
}  // namespace aicpu
#endif
