/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <map>
#include <vector>

#include "common/common_test.h"
#include "include/backend/mem_reuse/abstract_dynamic_mem_pool.h"

namespace mindspore {
namespace device {
size_t MallocLinearMem(size_t size, DeviceMemPtr *addr) {
  // actual peak size need base addr is not zero
  static size_t base_addr = 1 << 30;
  *addr = (void *)base_addr;
  base_addr += size;
  return size;
}

MemBufAllocatorPtr GenerateMemBufAllocatorPtr(size_t block_size = (1 << 30)) {
  bool is_persistent = true;
  uint32_t stream_id = kDefaultStreamIndex;
  std::function<MemBlock *(size_t)> mem_block_expander = [&, is_persistent = is_persistent, stream_id = stream_id,
                                                          block_size = block_size](size_t size) -> MemBlock * {
    size_t alloc_size = std::max(block_size, size);
    void *addr = nullptr;
    MallocLinearMem(alloc_size, &addr);
    auto mem_block = new MemBlock(alloc_size, addr, stream_id);
    return mem_block;
  };
  std::function<bool(MemBlock *)> mem_block_cleaner = [&](MemBlock *mem_block) { return true; };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [&](size_t size, void *addr) { return size; };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [&](void *addr, const size_t size) {
    return size;
  };
  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer, true,
                                           is_persistent, stream_id);
}

class TestMemBufAllocator : public UT::Common {
 public:
  TestMemBufAllocator() = default;
  virtual ~TestMemBufAllocator() = default;
};

/// Feature: test brief info for MemBufAllocator.
/// Description: test brief ino.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_brief_info) {
  auto allocator = GenerateMemBufAllocatorPtr();
  const auto &brief_info = allocator->BriefInfo();
  EXPECT_EQ("Mem buf allocator, is persistent : 1, stream id : 0.", brief_info);
}

/// Feature: test actual peak size for MemBufAllocator.
/// Description: test actual peak size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_actual_peak_size) {
  auto allocator = GenerateMemBufAllocatorPtr();
  auto mem_buf1 = allocator->Malloc(1 << 10);
  auto mem_buf2 = allocator->Malloc(1 << 9);
  auto mem_buf3 = allocator->Malloc(1 << 10);
  allocator->Free(mem_buf2);

  EXPECT_EQ((1 << 11) + (1 << 9), allocator->ActualPeakSize());
  allocator->Free(mem_buf1);
  allocator->Free(mem_buf3);
}

/// Feature: test malloc for MemBufAllocator.
/// Description: test malloc.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_malloc) {
  auto allocator = GenerateMemBufAllocatorPtr();
  auto mem_buf = allocator->Malloc(1 << 10);
  EXPECT_EQ(mem_buf->mem_block_->ToJson(), allocator->mem_blocks_.front()->ToJson());
  allocator->Free(mem_buf);
}

/// Feature: test allocate and free for MemBufAllocator.
/// Description: test allocate free.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMemBufAllocator, test_allocate_free) {
  auto allocator = GenerateMemBufAllocatorPtr(6 * kDynamicMemAlignSize);
  // test forward merge
  std::vector<size_t> front_merge_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize};
  std::vector<MemBuf *> front_merge_mem_bufs;
  for (auto size : front_merge_sizes) {
    auto mem_buf = allocator->Malloc(size);
    (void)front_merge_mem_bufs.emplace_back(mem_buf);
  }
  EXPECT_EQ(front_merge_mem_bufs[0]->next_, front_merge_mem_bufs[1]);
  EXPECT_EQ(front_merge_mem_bufs[1]->next_, front_merge_mem_bufs[2]);
  EXPECT_EQ(front_merge_mem_bufs[2]->prev_, front_merge_mem_bufs[1]);
  EXPECT_EQ(front_merge_mem_bufs[1]->prev_, front_merge_mem_bufs[0]);

  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(front_merge_mem_bufs[0]);
  allocator->Free(front_merge_mem_bufs[1]);
  auto first_buf = front_merge_mem_bufs[2]->prev_;
  EXPECT_EQ(first_buf->size_, 3 * kDynamicMemAlignSize);
  EXPECT_EQ(first_buf->prev_, nullptr);
  allocator->Free(front_merge_mem_bufs[2]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  front_merge_mem_bufs.clear();

  // test backward merge
  std::vector<size_t> back_merge_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize};
  std::vector<MemBuf *> back_merge_mem_bufs;
  for (auto size : back_merge_sizes) {
    (void)back_merge_mem_bufs.emplace_back(allocator->Malloc(size));
  }
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(back_merge_mem_bufs[2]);
  allocator->Free(back_merge_mem_bufs[1]);
  auto last_buf = back_merge_mem_bufs[0]->next_;
  EXPECT_EQ(last_buf->size_, 5 * kDynamicMemAlignSize);
  EXPECT_EQ(last_buf->next_, nullptr);
  allocator->Free(back_merge_mem_bufs[0]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  back_merge_mem_bufs.clear();

  // test forward and backward merge
  std::vector<size_t> merge_sizes{3 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 1 * kDynamicMemAlignSize};
  std::vector<MemBuf *> merge_mem_bufs;
  for (auto size : merge_sizes) {
    (void)merge_mem_bufs.emplace_back(allocator->Malloc(size));
  }
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  allocator->Free(merge_mem_bufs[2]);
  allocator->Free(merge_mem_bufs[0]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
  allocator->Free(merge_mem_bufs[1]);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ((*allocator->free_mem_bufs_.begin())->size_, 6 * kDynamicMemAlignSize);
  merge_mem_bufs.clear();
}

class LinearDynamicMemPool : public AbstractDynamicMemPool {
 public:
  LinearDynamicMemPool() { SetEnableVmm(true); }
  ~LinearDynamicMemPool() override = default;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override {
    static size_t base_addr = 0;
    *addr = (void *)base_addr;
    alloc_infos_[*addr] = size;
    base_addr += size;
    return size;
  }

  bool FreeDeviceMem(const DeviceMemPtr &addr) override {
    auto &&iter = alloc_infos_.find(addr);
    if (iter == alloc_infos_.end()) {
      return false;
    }
    alloc_infos_.erase(iter);
    return true;
  }

  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr) override {
    vmm_mmap_size_ += size;
    return size;
  }

  size_t GetMaxUsedMemSize() const override { return SIZE_MAX; }

  size_t GetVmmUsedMemSize() const override { return vmm_mmap_size_; }

  size_t free_mem_size() override { return SIZE_MAX; }

  uint64_t total_mem_size() const override {
    // when enable eager free, this size effect block size, use small size.
    return 1 << 30;
  }

  size_t ReservedMemorySize() {
    size_t reserved_memory_size = 0;
    for (auto &alloc_info : alloc_infos_) {
      reserved_memory_size += alloc_info.second;
    }
    return reserved_memory_size;
  }

 protected:
  // The related interface of device memory eager free.
  const bool IsEnableEagerFree() const override { return false; }

  const bool SyncAllStreams() override { return true; }
  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) override { return AllocDeviceMem(size, addr); }

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override { return size; }

 private:
  size_t vmm_mmap_size_{0};
  std::map<void *, size_t> alloc_infos_;
};

class TestAbstractDynamicMemPool : public UT::Common {
 public:
  TestAbstractDynamicMemPool() = default;
  virtual ~TestAbstractDynamicMemPool() = default;
};

/// Feature: test basic memory allocation for abstract dynamic mem pool.
/// Description: test basic allocation.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_basic_allocation) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  // Malloc 512M
  EXPECT_EQ(mem_pool->TotalMemStatistics(), 0);

  auto addr1 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr1, 0);

  // Malloc 512M
  auto addr2 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr2, kGBToByte / 2);

  // Malloc more 1g from persistent pool
  auto addr3 = mem_pool->AllocTensorMem(kGBToByte);
  EXPECT_EQ((size_t)addr3, kGBToByte);

  // Malloc more 1g
  auto addr4 = mem_pool->AllocTensorMem(kGBToByte);
  EXPECT_EQ((size_t)addr4, kGBToByte * 2);

  // Malloc another 512M
  auto addr5 = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr5, kGBToByte * 3);

  mem_pool->FreeTensorMem(addr1);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);

  // Malloc another 512M
  auto another_addr = mem_pool->AllocTensorMem(kGBToByte / 2);
  EXPECT_EQ((size_t)addr1, (size_t)another_addr);

  mem_pool->FreeTensorMem(addr4);
  // Malloc another 512M
  another_addr = mem_pool->AllocTensorMem(kGBToByte / 2, false, true);
  EXPECT_EQ((size_t)addr4, (size_t)another_addr);

  // Free 512M from common pool
  mem_pool->FreeTensorMem(addr2);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
}

/// Feature: test alloc aligned continuous tensor mem for abstract dynamic mem pool.
/// Description: test alloc continuous tensor mem with aligned size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_alloc_continuous_tensor_mem_with_aligned_size) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<size_t> alignd_sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize,
                                   4 * kDynamicMemAlignSize};
  const auto &aligned_addresses = mem_pool->AllocContinuousTensorMem(alignd_sizes, kDefaultStreamIndex);
  EXPECT_EQ(aligned_addresses.size(), alignd_sizes.size());
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[0]) + kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[1]));
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[1]) + 2 * kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[2]));
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[2]) + 3 * kDynamicMemAlignSize,
            reinterpret_cast<size_t>(aligned_addresses[3]));
  // assert mem buf counts is 4
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), (size_t)1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), alignd_sizes.size());
  // malloc more mem buf
  void *addr = mem_pool->AllocTensorMem(1);
  EXPECT_EQ(reinterpret_cast<size_t>(aligned_addresses[3]) + 4 * kDynamicMemAlignSize, reinterpret_cast<size_t>(addr));
  // free first two address
  mem_pool->FreeTensorMem(aligned_addresses[0]);
  mem_pool->FreeTensorMem(aligned_addresses[1]);
  // malloc again to assert reust first two address
  void *first_addr = mem_pool->AllocTensorMem(alignd_sizes[0]);
  void *second_addr = mem_pool->AllocTensorMem(alignd_sizes[1]);
  EXPECT_EQ(aligned_addresses[0], first_addr);
  EXPECT_EQ(aligned_addresses[1], second_addr);
  // free used addresses
  mem_pool->FreeTensorMem(addr);
  mem_pool->FreeTensorMem(first_addr);
  mem_pool->FreeTensorMem(second_addr);
  mem_pool->FreeTensorMem(aligned_addresses[2]);
  mem_pool->FreeTensorMem(aligned_addresses[3]);
  // assert free mem buf size is 1
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 0);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
}

/// Feature: test alloc unaligned continuous tensor mem for abstract dynamic mem pool.
/// Description: test alloc continuous tensor mem with unaligned size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_alloc_continuous_tensor_mem_with_unalignd_size) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<size_t> sizes{1, 2, 3, 4};
  const auto &addresses = mem_pool->AllocContinuousTensorMem(sizes, kDefaultStreamIndex);
  EXPECT_EQ(addresses.size(), sizes.size());
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[0]) + 1, reinterpret_cast<size_t>(addresses[1]));
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[1]) + 2, reinterpret_cast<size_t>(addresses[2]));
  EXPECT_EQ(reinterpret_cast<size_t>(addresses[2]) + 3, reinterpret_cast<size_t>(addresses[3]));
  // assert mem buf counts is 4.
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), sizes.size());
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 0);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  auto mem_buf = *(allocator->eager_free_mem_bufs_.begin());
  auto four_mem_buf = mem_buf->prev_;
  EXPECT_EQ(four_mem_buf->size_, 4);
  auto three_mem_buf = four_mem_buf->prev_;
  EXPECT_EQ(three_mem_buf->size_, 3);
  auto two_mem_buf = three_mem_buf->prev_;
  EXPECT_EQ(two_mem_buf->size_, 2);
  auto one_mem_buf = two_mem_buf->prev_;
  EXPECT_EQ(one_mem_buf->size_, 1);
}

/// Feature: test free part tensor mem keep first part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep first part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_first_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *first_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> first_free_addrs;
  (void)first_free_addrs.emplace_back(first_addr);
  std::vector<void *> first_keep_addrs;
  (void)first_keep_addrs.emplace_back(first_addr);
  std::vector<size_t> first_keep_addr_sizes;
  (void)first_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(first_free_addrs, first_keep_addrs, first_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
}

/// Feature: test free part tensor mem keep middle part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep middle part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_middle_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *middle_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> middle_free_addrs;
  (void)middle_free_addrs.emplace_back(middle_addr);
  std::vector<void *> middle_keep_addrs;
  (void)middle_keep_addrs.emplace_back(static_cast<int8_t *>(middle_addr) + kDynamicMemAlignSize);
  std::vector<size_t> middle_keep_addr_sizes;
  (void)middle_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(middle_free_addrs, middle_keep_addrs, middle_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 2);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 1);
}

/// Feature: test free part tensor mem keep last part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep last part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_last_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *last_addr = mem_pool->AllocTensorMem(3 * kDynamicMemAlignSize, true, true);
  std::vector<void *> last_free_addrs;
  (void)last_free_addrs.emplace_back(last_addr);
  std::vector<void *> last_keep_addrs;
  (void)last_keep_addrs.emplace_back(static_cast<int8_t *>(last_addr) + 2 * kDynamicMemAlignSize);
  std::vector<size_t> last_keep_addr_sizes;
  (void)last_keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(last_free_addrs, last_keep_addrs, last_keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 1);
}

/// Feature: test free part tensor mem keep multi part for abstract dynamic mem pool.
/// Description: test free part tensor mems keep multi part.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_free_part_tensor_mems_keep_multi_part) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  void *addr = mem_pool->AllocTensorMem(4 * kDynamicMemAlignSize, true, true);
  std::vector<void *> free_addrs;
  (void)free_addrs.emplace_back(addr);
  std::vector<void *> keep_addrs;
  (void)keep_addrs.emplace_back(addr);
  (void)keep_addrs.emplace_back(static_cast<int8_t *>(addr) + 2 * kDynamicMemAlignSize);
  (void)keep_addrs.emplace_back(static_cast<int8_t *>(addr) + 3 * kDynamicMemAlignSize);
  std::vector<size_t> keep_addr_sizes;
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  (void)keep_addr_sizes.emplace_back(kDynamicMemAlignSize);
  mem_pool->FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  EXPECT_EQ(mem_pool->stream_id_allocators_.size(), 1);
  auto allocator = mem_pool->stream_id_allocators_.begin()->second;
  EXPECT_EQ(allocator->mem_blocks_.size(), 1);
  EXPECT_EQ(allocator->free_mem_bufs_.size(), 1);
  EXPECT_EQ(allocator->eager_free_mem_bufs_.size(), 1);
  EXPECT_EQ(mem_pool->addr_mem_buf_allocators_.size(), 3);
}

/// Feature: test memory stats for abstract dynamic mem pool.
/// Description: test memory stats.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_memory_stats) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  auto &&block_counts = mem_pool->BlockCountsStatistics();
  EXPECT_EQ(block_counts.size(), 2);
  EXPECT_EQ(block_counts.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(block_counts[kPersistentMemPoolType], 0);
  EXPECT_EQ(block_counts.count(kCommonMemPoolType), 1);
  EXPECT_EQ(block_counts[kCommonMemPoolType], 0);

  auto &&block_unit_size = mem_pool->BlockUnitSizeStatistics();
  EXPECT_EQ(block_unit_size.size(), 2);
  EXPECT_EQ(block_unit_size.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(block_unit_size[kPersistentMemPoolType], kGBToByte);
  EXPECT_EQ(block_unit_size.count(kCommonMemPoolType), 1);
  EXPECT_EQ(block_unit_size[kPersistentMemPoolType], kGBToByte);

  auto &&common_mem_blocks_info = mem_pool->CommonMemBlocksInfoStatistics();
  EXPECT_EQ(common_mem_blocks_info.size(), 0);

  auto &&persistent_mem_blocks_info = mem_pool->PersistentMemBlocksInfoStatistics();
  EXPECT_EQ(persistent_mem_blocks_info.size(), 0);

  std::vector<size_t> sizes{1 * kDynamicMemAlignSize, 2 * kDynamicMemAlignSize, 3 * kDynamicMemAlignSize,
                            4 * kDynamicMemAlignSize, 5 * kDynamicMemAlignSize, 6 * kDynamicMemAlignSize};
  std::vector<void *> addresses;
  for (auto size : sizes) {
    (void)addresses.emplace_back(mem_pool->AllocTensorMem(size));
  }
  auto &&cur_block_counts = mem_pool->BlockCountsStatistics();
  EXPECT_EQ(cur_block_counts.size(), 2);
  EXPECT_EQ(cur_block_counts.count(kPersistentMemPoolType), 1);
  EXPECT_EQ(cur_block_counts[kPersistentMemPoolType], 0);
  EXPECT_EQ(cur_block_counts.count(kCommonMemPoolType), 1);
  EXPECT_EQ(cur_block_counts[kCommonMemPoolType], 1);
  // assert block count.
  EXPECT_EQ(mem_pool->CommonMemBlocksInfoStatistics().size(), 1);
  EXPECT_EQ(mem_pool->PersistentMemBlocksInfoStatistics().size(), 0);
}

/// Feature: test memory reserved size for abstract dynamic mem pool.
/// Description: test memory reserved size.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAbstractDynamicMemPool, test_memory_reserved_dize) {
  auto mem_pool = std::make_shared<LinearDynamicMemPool>();
  std::vector<void *> addrs;
  for (size_t i = 0; i < 100; i++) {
    (void)addrs.emplace_back(mem_pool->AllocTensorMem(i));
  }
  EXPECT_EQ(mem_pool->ReservedMemorySize(), kGBToByte);
  mem_pool->ReleaseDeviceRes();
  EXPECT_EQ(mem_pool->ReservedMemorySize(), 0);
}
}  // namespace device
}  // namespace mindspore
