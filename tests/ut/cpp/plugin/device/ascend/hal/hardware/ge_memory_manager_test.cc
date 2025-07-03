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
#include "common/common_test.h"
#define private public
#define protected public
#include "backend/ge_backend/executor/ge_memory_manager.h"
#undef private
#undef protected

namespace mindspore {
namespace backend {
namespace ge_backend {
class TestGeMemoryManager : public UT::Common {
 public:
  TestGeMemoryManager() = default;
  virtual ~TestGeMemoryManager() = default;

  void SetUp() override { GEMemoryManager::Instance().Clear(); }
  void TearDown() override { GEMemoryManager::Instance().Clear(); }
};

/// Feature: test ge memory manager.
/// Description: test init ge memory.
/// Expectation: can init ge memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_init_ge_memory) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().graph_memory_.find("test"), GEMemoryManager::Instance().graph_memory_.end());
  auto ge_memory = GEMemoryManager::Instance().graph_memory_["test"];
  EXPECT_EQ(ge_memory.run_options.name, "test");
  EXPECT_EQ(ge_memory.workspace_memory, 1024);
  EXPECT_EQ(ge_memory.fixed_memory, 1024);
  EXPECT_EQ(ge_memory.const_memory, 1024);
  EXPECT_EQ(ge_memory.is_refreshable, true);
  EXPECT_EQ(ge_memory.stream_id, 0);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_graphs_.find(0),
            GEMemoryManager::Instance().stream_id_to_graphs_.end());
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_graphs_[0].size(), 1);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(0),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[0], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[0]->memory_size, 1024);
}

/// Feature: test ge memory manager.
/// Description: test reuse ge memory.
/// Expectation: can reuse ge memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_reuse_ge_memory) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(0),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[0], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[0]->memory_size, 1024);
  options.name = "test1";
  fixed_memory_size = 2048;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(0),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[0], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[0]->memory_size, 2048);
  auto fix_memory_ptrs = GEMemoryManager::Instance().GetAllNotAllocFixMemory();
  EXPECT_EQ(fix_memory_ptrs.size(), 1);
}

/// Feature: test ge memory manager.
/// Description: test no reuse ge memory multi stream.
/// Expectation: no reuse ge memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_no_reuse_ge_memory_multi_stream) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(0),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[0], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[0]->memory_size, 1024);
  options.name = "test1";
  fixed_memory_size = 2048;
  stream_id = 1;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(1),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[1], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[1]->memory_size, 2048);
  auto fix_memory_ptrs = GEMemoryManager::Instance().GetAllNotAllocFixMemory();
  EXPECT_EQ(fix_memory_ptrs.size(), 2);
}

/// Feature: test ge memory manager.
/// Description: test no reuse ge memory after alloc.
/// Expectation: no reuse ge memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_no_reuse_ge_memory_after_alloc) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  size_t tmp_memory = 0;
  auto alloc_func = [&tmp_memory](size_t size) -> void * { return &tmp_memory; };
  auto update_func = [](bool is_refreshable, const backend::ge_backend::RunOptions &options, const void *const memory,
                        size_t size) -> backend::ge_backend::Status { return backend::ge_backend::Status::SUCCESS; };
  GEMemoryManager::Instance().AllocGEMemory(alloc_func, update_func);
  auto fix_memory_ptrs = GEMemoryManager::Instance().GetAllNotAllocFixMemory();
  EXPECT_EQ(fix_memory_ptrs.size(), 0);

  options.name = "test0";
  fixed_memory_size = 2048;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_.find(0),
            GEMemoryManager::Instance().stream_id_to_fix_memory_.end());
  EXPECT_NE(GEMemoryManager::Instance().stream_id_to_fix_memory_[0], nullptr);
  EXPECT_EQ(GEMemoryManager::Instance().stream_id_to_fix_memory_[0]->memory_size, 2048);
  fix_memory_ptrs = GEMemoryManager::Instance().GetAllNotAllocFixMemory();
  EXPECT_EQ(fix_memory_ptrs.size(), 1);

  EXPECT_NE(GEMemoryManager::Instance().graph_memory_.find("test"), GEMemoryManager::Instance().graph_memory_.end());
  auto ge_memory_test = GEMemoryManager::Instance().graph_memory_["test"];
  EXPECT_NE(GEMemoryManager::Instance().graph_memory_.find("test0"), GEMemoryManager::Instance().graph_memory_.end());
  auto ge_memory_test0 = GEMemoryManager::Instance().graph_memory_["test0"];
  EXPECT_NE(ge_memory_test.reuse_memory, ge_memory_test0.reuse_memory);
  EXPECT_EQ(ge_memory_test.reuse_memory->memory_size, 1024);
  EXPECT_EQ(ge_memory_test0.reuse_memory->memory_size, 2048);
}

/// Feature: test ge memory manager.
/// Description: test alloc ge memory.
/// Expectation: can alloc ge memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_alloc_ge_memory) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  size_t tmp_memory = 0;
  auto alloc_func = [&tmp_memory](size_t size) -> void * { return &tmp_memory; };
  auto update_func = [](bool is_refreshable, const backend::ge_backend::RunOptions &options, const void *const memory,
                        size_t size) -> backend::ge_backend::Status { return backend::ge_backend::Status::SUCCESS; };
  GEMemoryManager::Instance().AllocGEMemory(alloc_func, update_func);
  auto fix_memory_ptrs = GEMemoryManager::Instance().GetAllNotAllocFixMemory();
  EXPECT_EQ(fix_memory_ptrs.size(), 0);
}

/// Feature: test ge memory manager.
/// Description: test get workspace memory.
/// Expectation: can get workspace memory and can not throw exception.
TEST_F(TestGeMemoryManager, test_get_workspace) {
  size_t stream_id = 0;
  backend::ge_backend::RunOptions options;
  options.name = "test";
  size_t workspace_memory_size = 1024;
  size_t fixed_memory_size = 1024;
  size_t const_memory_size = 1024;
  size_t is_refreshable = true;
  GEMemoryManager::Instance().InitGEMemory(options, workspace_memory_size, fixed_memory_size, const_memory_size,
                                           is_refreshable, stream_id);
  auto workspace_memory = GEMemoryManager::Instance().GetWorkspaceMemory("test");
  EXPECT_EQ(workspace_memory, 1024);
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
