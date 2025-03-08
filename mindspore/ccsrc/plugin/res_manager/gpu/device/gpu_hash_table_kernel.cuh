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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_

#if CUDA_VERSION > 11000
#include <cuco/dynamic_map.cuh>
#include <curand_kernel.h>
#include "plugin/res_manager/gpu/device/gpu_hash_table_common.h"

namespace mindspore {
namespace device {
namespace gpu {
namespace cg = cooperative_groups;

// Check whether the key exist in map already.
template <typename CG, typename Key, typename View>
__device__ __forceinline__ void CheckKeyExist(const CG &g, const Key &key, View *submap_views, size_t submaps_num,
                                              int32_t *index_in_block) {
  for (size_t i = 0; i < submaps_num; ++i) {
    auto &submap_view = submap_views[i];
    auto iter = submap_view.find(g, key);
    if (iter != submap_view.end()) {
      *index_in_block = iter->second;
      break;
    }
  }
}

// Get a valid position in block and return the offset index.
template <typename CG>
__device__ __forceinline__ int32_t GetInsertIndex(const CG &g, const int32_t *erased_slot,
                                                  CudaAtomicInt *erased_counter, CudaAtomicSize *current_index) {
  int32_t candidate_index = 0;
  if (g.thread_rank() == 0) {
    if (erased_counter->load(cuda::std::memory_order_relaxed) > 0) {
      // Idle slot position is preferred.
      int32_t idle_index = erased_counter->fetch_sub(1, cuda::std::memory_order_relaxed) - 1;
      // Idle slot position compete fail.
      if (idle_index < 0) {
        erased_counter->store(0, cuda::std::memory_order_relaxed);
        candidate_index = current_index->fetch_add(1, cuda::std::memory_order_relaxed);
      } else {
        candidate_index = erased_slot[idle_index];
      }
    } else {
      // If idle slot is empty, use new position in blocks.
      candidate_index = current_index->fetch_add(1, cuda::std::memory_order_relaxed);
    }
  }
  // Sync index in block in cooperative group.
  int32_t index_in_block = g.shfl(candidate_index, 0);
  return index_in_block;
}

// Transform all keys into indices in blocks. If the key exist in map already ,just return the index,
// otherwise find a valid position in block.
template <uint32_t block_size, uint32_t tile_size, typename Key, typename MutableView, typename View>
__global__ void LookupIndices(const Key *keys, size_t key_num, bool insert_miss_key, size_t submaps_num,
                              size_t submap_idx, const int32_t *erased_slot, CudaAtomicInt *erased_counter,
                              MutableView *submap_mutable_views, View *submap_views, CudaAtomicSize *insert_success_num,
                              CudaAtomicSize *current_index, int32_t *indices) {}

// Initialize normal distribution random generator states.
__global__ void InitNormalDisRandomGen(uint32_t seed, curandStatePhilox4_32_10_t *state) {
  int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, global_thread_index, 0, &state[global_thread_index]);
}

// Insert default normal distribution random value.
template <typename Value>
__global__ void InsertNormalDistRandomValue(size_t value_dim, size_t total_insert_elements_num, const int *indices,
                                            size_t elements_per_block, size_t *const *lookup_cnts_ptr,
                                            size_t permit_threshold, const Value mean, const Value stddev,
                                            curandStatePhilox4_32_10_t *state, bool **idle_flags_ptr,
                                            Value *const *blocks_ptr) {}

// Insert default value into map by specific value.
template <typename Value>
__global__ void InsertDefaultValue(size_t value_dim, size_t total_insert_elements_num, const int *indices,
                                   size_t elements_per_block, size_t *const *lookup_cnts_ptr, size_t permit_threshold,
                                   const Value default_value, bool **idle_flags_ptr, Value *const *blocks_ptr) {}

// Get all values by indices in blocks.
template <typename Value>
__global__ void GetValues(size_t value_dim, size_t total_size, const int *indices, const size_t elements_per_block,
                          Value *const *blocks_ptr, Value *outputs) {}

// Insert values into map by indices in blocks.
template <typename Value>
__global__ void InsertValues(size_t value_dim, size_t total_insert_size, const int *indices, const Value *insert_values,
                             const size_t elements_per_block, const size_t *const *lookup_cnts_ptr,
                             size_t permit_threshold, size_t global_timestamp, size_t *const *update_timestamps_ptr,
                             Status *const *statuses_ptr, bool *const *idle_flags_ptr, Value *const *blocks_ptr) {}

template <uint32_t block_size>
__global__ void CountPermissionNum(size_t elements_per_block, size_t key_num, const int *indices,
                                   size_t *const *lookup_cnts_ptr, size_t permit_threshold,
                                   CudaAtomicSize *insert_counter) {}

template <uint32_t block_size>
__global__ void CountExpiredNum(size_t blocks_num, size_t permit_threshold, size_t global_timestamp,
                                uint64_t evict_threshold, size_t elements_per_block, bool *const *idle_flags_ptr,
                                size_t *const *lookup_cnts_ptr, size_t *const *update_timestamps_ptr,
                                CudaAtomicSize *expired_counter) {}

template <typename Key>
__global__ void FindExpiredKeysAndIndices(size_t key_num, size_t elements_per_block, size_t permit_threshold,
                                          size_t global_timestamp, uint64_t evict_threshold,
                                          bool *const *idle_flags_ptr, size_t *const *lookup_cnts_ptr,
                                          size_t *const *update_timestamps_ptr, const Key *all_keys,
                                          const int *all_indices, CudaAtomicSize *expired_counter, Key *expired_keys,
                                          int *expired_indices) {}

// Erase elements in hash map, update idle status for erased slots.
__global__ void EraseElementsByIndices(size_t erase_num, size_t elements_per_block, const int *erased_indices,
                                       bool *const *idle_flags_ptr, Status *const *statuses_ptr) {}

__global__ void AddErasedSlots(size_t erased_num, const int *erased_indices, CudaAtomicInt *erased_counter,
                               int *erased_slot) {}

// Update status of element in hash table.
__global__ void UpdateStatus(size_t key_num, size_t elements_per_block, int empty_index, const int *indices,
                             Status new_status, Status *const *statuses_ptr) {}

// Count the number of element whose statuses are modified.
template <uint32_t block_size>
__global__ void CountModifiedNum(size_t blocks_num, size_t elements_per_block, const Status *const *statuses_ptr,
                                 CudaAtomicSize *modified_counter) {}

// Find all keys and indices for elememts whose statuses are modified.
template <typename Key>
__global__ void FindModifiedKeysAndIndices(size_t key_num, size_t elements_per_block, const Key *all_keys,
                                           const int *all_indices, const Status *const *statuses_ptr,
                                           CudaAtomicSize *modified_counter, Key *modified_keys,
                                           int *modified_indices) {}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_
