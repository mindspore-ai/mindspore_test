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
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm/hccl_watch_dog_thread.h"

namespace mindspore {
namespace device {
namespace ascend {
class TestHcclWatchDogHandler : public UT::Common {
 public:
  TestHcclWatchDogHandler() = default;
  virtual ~TestHcclWatchDogHandler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

class TestHcclWatchDogManager : public UT::Common {
 public:
  TestHcclWatchDogManager() = default;
  virtual ~TestHcclWatchDogManager() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: HcclWatchDogHandler.
/// Description: Test HcclWatchDogHandler initialize.
/// Expectation: None
TEST_F(TestHcclWatchDogHandler, initialize1) {
  uint32_t global_rank_id = 0;
  uint32_t local_rank_id = 0;
  uint32_t global_rank_size = 8;
  std::map<std::string, HcclComm> mp_;
  auto handler = std::make_shared<HcclWatchDogHandler>(global_rank_id, local_rank_id, global_rank_size, mp_);
  EXPECT_EQ(handler->global_rank(), global_rank_id);
  EXPECT_EQ(handler->local_rank(), local_rank_id);
  EXPECT_EQ(handler->global_rank_size(), global_rank_size);
}

/// Feature: HcclWatchDogHandler.
/// Description: Test HcclWatchDogHandler initialize2, create thread.
/// Expectation: None
TEST_F(TestHcclWatchDogHandler, initialize2) {
  uint32_t global_rank_id = 0;
  uint32_t local_rank_id = 0;
  uint32_t global_rank_size = 8;
  std::map<std::string, HcclComm> mp_;
  auto handler = std::make_shared<HcclWatchDogHandler>(global_rank_id, local_rank_id, global_rank_size, mp_);
  EXPECT_EQ(handler->Initialize(), true);
}

/// Feature: TestHcclWatchDogManager.
/// Description: Test HcclWatchDogManager init, no handle, return false.
/// Expectation: None
TEST_F(TestHcclWatchDogManager, InitializeWithoutHandle) {
  EXPECT_EQ(HcclWatchDogManager::GetInstance().InitHandler(), false);
}

/// Feature: TestHcclWatchDogManager.
/// Description: Test HcclWatchDogManager init, with handle, return true.
/// Expectation: None
TEST_F(TestHcclWatchDogManager, InitializeWithHandle) {
  uint32_t global_rank_id = 0;
  uint32_t local_rank_id = 0;
  uint32_t global_rank_size = 8;
  std::map<std::string, HcclComm> mp_;
  HcclWatchDogManager::GetInstance().AddHandler(
    std::make_unique<HcclWatchDogHandler>(global_rank_id, local_rank_id, global_rank_size, mp_));
  EXPECT_EQ(HcclWatchDogManager::GetInstance().InitHandler(), true);
}

/// Feature: TestHcclWatchDogManager.
/// Description: Test HcclWatchDogManager init, no handle, no exception.
/// Expectation: None
TEST_F(TestHcclWatchDogManager, DestroyWithoutHandle) {
  EXPECT_NO_THROW(HcclWatchDogManager::GetInstance().DestoryHandler());
}

/// Feature: TestHcclWatchDogManager.
/// Description: Test destroy HcclWatchDogManager, with handle, no exception.
/// Expectation: None
TEST_F(TestHcclWatchDogManager, DestroyWithHandle) {
  uint32_t global_rank_id = 0;
  uint32_t local_rank_id = 0;
  uint32_t global_rank_size = 8;
  std::map<std::string, HcclComm> mp_;
  HcclWatchDogManager::GetInstance().AddHandler(
    std::make_unique<HcclWatchDogHandler>(global_rank_id, local_rank_id, global_rank_size, mp_));
  EXPECT_NO_THROW(HcclWatchDogManager::GetInstance().DestoryHandler());
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
