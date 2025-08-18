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

#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "common/common_test.h"
#include "ir/device_address.h"

namespace mindspore {
namespace runtime {
class OpsCommonTest : public UT::Common {
 public:
  OpsCommonTest() {}
};

/// Feature: Test device address size.
/// Description: Test device address size.
/// Expectation: As expected.
TEST_F(OpsCommonTest, CalDeviceAddressSize) {
  size_t device_address_size = sizeof(device::DeviceAddress);
  size_t expected_size = 72;
  ASSERT_TRUE(device_address_size <= expected_size);
}
}  // namespace runtime
}  // namespace mindspore
