/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "utils/ms_exception.h"
#include "utils/ms_utils.h"

namespace mindspore {
class TestUCEException : public UT::Common {
 public:
  TestUCEException() = default;
  virtual ~TestUCEException() = default;

  void SetUp() override {}
  void TearDown() override {}
};


/// Feature: test UCE Exception instance.
/// Description:Test all interfaces in the instance.
/// Expectation: The return value of the normal interface is as expected, and the exception branch is expected to catch
/// the exception.
TEST_F(TestUCEException, test_interface) {
  EXPECT_EQ(UCEException::IsEnableUCE(), false);

  const auto kTftEnv = "MS_ENABLE_TFT";
  common::SetEnv(kTftEnv, "{TTP:1,UCE:1,ARF:1}");
  EXPECT_NO_THROW(UCEException::GetInstance().CheckUceARFEnv());
  EXPECT_EQ(UCEException::IsEnableUCE(), false);
  EXPECT_EQ(UCEException::GetInstance().get_has_throw_error(), false);
  EXPECT_EQ(UCEException::GetInstance().get_force_stop_flag(), false);
  EXPECT_EQ(UCEException::GetInstance().get_uce_flag(), false);
  EXPECT_EQ(UCEException::GetInstance().is_reboot_node(), false);
  EXPECT_EQ(UCEException::GetInstance().is_arf(), false);

  EXPECT_NO_THROW(UCEException::GetInstance().set_is_arf(true));
  EXPECT_NO_THROW(UCEException::GetInstance().set_reboot_node(true));
  EXPECT_EQ(UCEException::GetInstance().is_reboot_node(), true);
  EXPECT_EQ(UCEException::GetInstance().is_arf(), true);
  EXPECT_EQ(UCEException::GetInstance().get_has_throw_error(), true);

  EXPECT_NO_THROW(UCEException::GetInstance().set_force_stop_flag(true));
  EXPECT_EQ(UCEException::GetInstance().get_force_stop_flag(), true);
}
}  // namespace mindspore
