/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
#define TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H

#include <memory>

#include "common/common_test.h"
#define private public
#define protected public
#include "abstract/abstract_function.h"
#include "runtime/graph_scheduler/control_node_parser.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "common/device_address.h"
#include "common/kernel_tensor.h"
#include "common/kernel_utils.h"
#include "common/common_utils.h"
#include "kernel/framework_utils.h"
#include "runtime/device/res_manager/test_device_address.h"
#define private public
#define protected public

namespace mindspore {
namespace runtime {
namespace test {
using abstract::AbstractFuncUnion;
using abstract::AbstractTensor;
using abstract::AbstractTensorPtr;
using abstract::AnalysisContext;
using abstract::FuncGraphAbstractClosure;
using device::DeviceAddress;
using device::DeviceAddressPtr;
using device::DeviceContextKey;
using device::DeviceContextRegister;
using device::DeviceType;
using kernel::AddressPtr;
using kernel::KernelTensorPtr;
using session::KernelGraph;
}  // namespace test
}  // namespace runtime
}  // namespace mindspore
#endif  // TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
