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

#ifndef TESTS_UT_CPP_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_COMMON_TEST_SUITE_H_
#define TESTS_UT_CPP_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_COMMON_TEST_SUITE_H_

#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "op_def/framework_ops.h"
#include "backend/common/graph_kernel/kernel_packet/symbol_engine_extender.h"
#include "backend/common/graph_kernel/convert_call_to_prim.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel::test {
class TestKernelPacket : public GraphKernelCommonTestSuite {};
}  // namespace mindspore::graphkernel::test
#endif  // TESTS_UT_CPP_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_COMMON_TEST_SUITE_H_
