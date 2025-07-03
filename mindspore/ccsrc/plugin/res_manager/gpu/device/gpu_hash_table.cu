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

#include "plugin/res_manager/gpu/device/gpu_hash_table.h"

#if CUDA_VERSION > 11000
#include <cuco/dynamic_map.cuh>
#include <random>
#include <algorithm>
#include <unordered_set>

#include "plugin/res_manager/gpu/device/gpu_hash_table_kernel.cuh"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "plugin/res_manager/gpu/device/gpu_device_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
template <typename Key, typename Value, typename Allocator>
using CucoDynamicMap = cuco::dynamic_map<Key, Value, cuda::thread_scope_device, Allocator>;

// CudaDynamicMap is a wrapper of cuco::dynamic_map, gpu_hash_table.h needs to be used by other cpp source files, in
// order for g++ to compile properly, the declaration of the cuco::dynamic_map type cannot appear in the header file
// gpu_hash_table.h, through the CudaDynamicMap type gpu_hash_ table.h pre-declaration to solve compilation problems.
template <typename Key, typename Value, typename Allocator>
struct CudaDynamicMap {
  CucoDynamicMap<Key, Value, Allocator> dynamic_map_;

  CudaDynamicMap(const Key &empty_key, const Value &empty_value, const Key &erased_key, const Allocator &alloc,
                 cudaStream_t stream = 0)
      : dynamic_map_(kInitialCapacity, cuco::sentinel::empty_key<Key>{empty_key},
                     cuco::sentinel::empty_value<Value>{empty_value}, cuco::sentinel::erased_key<Key>{erased_key},
                     alloc, stream) {}

  ~CudaDynamicMap() = default;
};

template <typename Key, typename Value, typename Allocator>
std::vector<int8_t> GPUHashTable<Key, Value, Allocator>::idle_flags_initializer_ =
  std::vector<int8_t>(GPUHashTable<Key, Value, Allocator>::elements_per_block_, 1);

template <typename Key, typename Value, typename Allocator>
std::vector<size_t> GPUHashTable<Key, Value, Allocator>::lookup_counter_initializer_ =
  std::vector<size_t>(GPUHashTable<Key, Value, Allocator>::elements_per_block_, 0);

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const std::string &initializer,
                                                  uint64_t permit_threshold, uint64_t evict_threshold,
                                                  const Allocator &alloc)
    : value_dim_(value_dim),
      initializer_(initializer),
      default_value_(0),
      char_alloc_(alloc),
      permit_threshold_(permit_threshold),
      evict_threshold_(evict_threshold) {
  Initialize(alloc);
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const Value &default_value,
                                                  uint64_t permit_threshold, uint64_t evict_threshold,
                                                  const Allocator &alloc)
    : value_dim_(value_dim),
      initializer_(""),
      default_value_(default_value),
      char_alloc_(alloc),
      permit_threshold_(permit_threshold),
      evict_threshold_(evict_threshold) {
  Initialize(alloc);
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::~GPUHashTable() {
  Finalize();
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::Initialize(const Allocator &alloc) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  cuda_dynamic_map_ = std::make_unique<CudaDynamicMap<Key, int32_t, Allocator>>(
    static_cast<Key>(kEmptyKey), kEmptyValue, static_cast<Key>(kErasedKey), alloc, stream);

  CudaAtomicSize host_init_atomic_size_t(0);
  CudaAtomicInt host_init_atomic_int(0);

  AllocateMemory(sizeof(CudaAtomicSize), &current_index_);
  AllocateMemory(sizeof(CudaAtomicInt), &erased_counter_);

  CHECK_CUDA_RET(
    cudaMemcpyAsync(current_index_, &host_init_atomic_size_t, sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");
  CHECK_CUDA_RET(
    cudaMemcpyAsync(erased_counter_, &host_init_atomic_int, sizeof(CudaAtomicInt), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");

  CHECK_CUDA_RET(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  CHECK_CUDA_RET(cudaMallocManaged(&insert_success_number_, sizeof(CudaAtomicSize)), "cudaMallocManaged");
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::Finalize() {
  cuda_dynamic_map_ = nullptr;

  FreeMemory(current_index_);
  FreeMemory(erased_counter_);

  if (erased_slot_) {
    FreeMemory(erased_slot_);
  }

  for (size_t i = 0; i < blocks_.size(); i++) {
    FreeMemory(blocks_[i]);
    FreeMemory(idle_flags_[i]);
    FreeMemory(statuses_[i]);
    FreeMemory(lookup_cnts_[i]);
    FreeMemory(update_timestamps_[i]);
  }

  FreeAllBlockRecorders();

  if (random_gen_state_) {
    FreeMemory(random_gen_state_);
  }

  CHECK_CUDA_RET(cudaFree(insert_success_number_), "cudaFree");
}

template <typename Key, typename Value, typename Allocator>
template <typename T>
void GPUHashTable<Key, Value, Allocator>::AllocateMemory(size_t size, T **ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  *ptr = reinterpret_cast<T *>(std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, size));
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::FreeMemory(void *ptr) {
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, reinterpret_cast<char *>(ptr), 0);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               Value *outputs, void *stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!initializer_.empty()) {
    return Find(keys, key_num, insert_default_value, initializer_, outputs, stream);
  }
  return Find(keys, key_num, insert_default_value, default_value_, outputs, stream);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               const std::string &initializer, Value *outputs, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               const Value &default_value, Value *outputs, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Insert(const Key *keys, size_t key_num, const Value *value, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Erase(const Key *keys, size_t key_num, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Clear() {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Reserve(size_t new_capacity, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::AddNewBlock(cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::ResetAllBlockRecorders(cudaStream_t cuda_stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::FreeAllBlockRecorders() {
  if (blocks_ptr_) {
    FreeMemory(blocks_ptr_);
  }
  if (idle_flags_ptr_) {
    FreeMemory(idle_flags_ptr_);
  }
  if (statuses_ptr_) {
    FreeMemory(statuses_ptr_);
  }
  if (lookup_cnts_ptr_) {
    FreeMemory(lookup_cnts_ptr_);
  }
  if (update_timestamps_ptr_) {
    FreeMemory(update_timestamps_ptr_);
  }
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetKeysAndValues(Key *keys, Value *values, void *stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::EvictExpiredElements(cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::CountExpiredElements(cudaStream_t stream, size_t *expired_num) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::FindExpiredElements(Key *expired_keys, int *expired_indices,
                                                              cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::EraseElements(const Key *keys, size_t key_num, const int *indices,
                                                        cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Import(const DataLenPair &input_data) {
  // 1. Store input tensor data until receiving kImportTensorNum(3) input tensor.
  // Really import input data to hash table when receive kImportTensorNum(3) input tensor.
  return true;
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::Export(bool incremental) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  // Evict expired element before export.
  MS_EXCEPTION_IF_CHECK_FAIL(EvictExpiredElements(stream), "Evict expired elements failed.");

  // Update is_dirty_ to false because host side will get latest content after export.
  is_dirty_ = false;

  if (incremental) {
    return ExportIncrementally(stream);
  }
  return ExportFully(stream);
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportSlice(bool incremental, bool *last_slice, size_t) {
  MS_EXCEPTION_IF_NULL(last_slice);

  *last_slice = true;
  auto ret = Export(incremental);
  is_dirty_ = true;
  return ret;
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportFully(cudaStream_t stream) {
  return {std::make_shared<std::vector<char>>(), std::make_shared<std::vector<char>>(),
          std::make_shared<std::vector<char>>()};
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportIncrementally(cudaStream_t stream) {
  return {std::make_shared<std::vector<char>>(), std::make_shared<std::vector<char>>(),
          std::make_shared<std::vector<char>>()};
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::CountModifiedElements(cudaStream_t stream, size_t *modified_num) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::FindModifiedElements(Key *modified_keys, int *modified_indices,
                                                               cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportModifiedAndErasedElements(
  size_t modified_num, const Key *device_modified_keys, const Value *device_modified_values, cudaStream_t stream) {
  return {std::make_shared<std::vector<char>>(), std::make_shared<std::vector<char>>(),
          std::make_shared<std::vector<char>>()};
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetIndicesByKeys(const Key *key, size_t key_num, bool insert_miss_key,
                                                           int32_t *indices, cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::UpdateSize(size_t key_num, const int *indices, cudaStream_t stream,
                                                     bool update_lookup_count) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::InsertDefaultValueByInitializer(size_t key_num,
                                                                          const std::string &initializer,
                                                                          const int *indices, cudaStream_t stream) {
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::InitNormalDistRandomGenerator(cudaStream_t stream) {
  return true;
}

template class GPUHashTable<int32_t, float>;
template class GPUHashTable<int64_t, float>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
