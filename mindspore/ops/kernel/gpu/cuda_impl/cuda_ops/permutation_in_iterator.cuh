/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PERMUTATION_IN_ITERATOR_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PERMUTATION_IN_ITERATOR_CUH_

#include <iostream>
#include <iterator>

template <typename ValueType,            ///< The value type of this iterator
          typename IndexIteratorT,       ///< Iterator for mapping objects of InputIterator
          typename InputIteratorT,       ///< The type of the wrapped input iterator
          typename OffsetT = ptrdiff_t>  ///< The difference type of this iterator (Default: ptrdiff_t)
class PermutationInputIterator {
 public:
  // Required iterator traits
  typedef PermutationInputIterator self_type;  ///< My own type
  typedef OffsetT difference_type;             ///< Type to express the result of subtracting one iterator from another

  typedef ValueType value_type;  ///< The type of the element the iterator can point to
  typedef ValueType *pointer;    ///< The type of a pointer to an element the iterator can point to
  typedef ValueType reference;   ///< The type of a reference to an element the iterator can point to

  typedef std::random_access_iterator_tag iterator_category;  ///< The iterator category

 private:
  InputIteratorT input_iterator;
  IndexIteratorT index_iterator;

 public:
  /// Constructor
  __host__ __device__ __forceinline__
  PermutationInputIterator(InputIteratorT input_iterator,  ///< Input iterator to wrap
                           IndexIteratorT index_iterator)  ///< Conversion iterator to warp
      : input_iterator(input_iterator), index_iterator(index_iterator) {}

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    index_iterator++;
    return *this;
  }

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type ret = *this;
    index_iterator++;
    return ret;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const { return input_iterator[*index_iterator]; }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    index_iterator += n;
    return *this;
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type ret(input_iterator, index_iterator + n);
    return ret;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(input_iterator, index_iterator - n);
    return retval;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_type other) const {
    return index_iterator - other.index_iterator;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator-=(Distance n) {
    index_iterator -= n;
    return *this;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    return input_iterator[index_iterator[n]];
  }

  /// Structure dereference
  __host__ __device__ __forceinline__ pointer operator->() { return input_iterator + *index_iterator; }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) {
    return (index_iterator == rhs.index_iterator && input_iterator == rhs.input_iterator);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) { return !(*this == rhs); }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type &itr) { return os; }
};

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PERMUTATION_IN_ITERATOR_CUH_
