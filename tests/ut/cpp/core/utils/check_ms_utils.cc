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
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "utils/ms_utils.h"

namespace mindspore {
class TestCheckMsUtils : public UT::Common {
 public:
  TestCheckMsUtils() = default;
  void SetUp() {}
};

// Feature: TestCheckMsUtils.
// Description: Check function of IsEnableRuntimeConfig and IsDisableRuntimeConfig in ms_utils.cc
// Expectation: Get right runtime config.
TEST_F(TestCheckMsUtils, test_read_runtime_config) {
  const char *test_configs1[] = {
    "inline:true,compile_statistic:True,memoty_statistic:false",
    "'inline:true,compile_statistic:True,memoty_statistic:false'",
    "\"inline:true,compile_statistic:True,memoty_statistic:false\"",
  };

  const char *test_configs2[] = {"all_finite:true, memoty_statistic:True, inline:false",
                                 "all_finite:true; memoty_statistic:True; inline:false",
                                 "all_finite:true;memoty_statistic:True;inline:false"};

  for (const auto &config : test_configs1) {
    int ret = common::SetEnv("MS_DEV_RUNTIME_CONF", config);
    ASSERT_EQ(ret, 0);

    ASSERT_TRUE(runtime::IsEnableRuntimeConfig("inline"));
    ASSERT_TRUE(runtime::IsEnableRuntimeConfig("compile_statistic"));
    ASSERT_FALSE(runtime::IsEnableRuntimeConfig("memoty_statistic"));
    ASSERT_TRUE(runtime::IsDisableRuntimeConfig("memoty_statistic"));
    ASSERT_FALSE(runtime::IsEnableRuntimeConfig("switch_inline"));
    ASSERT_FALSE(runtime::IsDisableRuntimeConfig("switch_inline"));

    (void)common::ResetConfig("MS_DEV_RUNTIME_CONF");
  }

  // Second group of tests
  for (const auto &config : test_configs2) {
    int ret = common::SetEnv("MS_DEV_RUNTIME_CONF", config);
    ASSERT_EQ(ret, 0);

    ASSERT_TRUE(runtime::IsEnableRuntimeConfig("all_finite"));
    ASSERT_TRUE(runtime::IsEnableRuntimeConfig("memoty_statistic"));
    ASSERT_FALSE(runtime::IsEnableRuntimeConfig("inline"));
    ASSERT_TRUE(runtime::IsDisableRuntimeConfig("inline"));
    ASSERT_FALSE(runtime::IsEnableRuntimeConfig("pipeline"));
    ASSERT_FALSE(runtime::IsDisableRuntimeConfig("pipeline"));
    (void)common::ResetConfig("MS_DEV_RUNTIME_CONF");
  }
}
}  // namespace mindspore
