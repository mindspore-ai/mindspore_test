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
#include "mindspore/ops/op_def/framework_ops.h"
#include "backend/common/graph_kernel/kernel_packet/symbol_engine_extender.h"
#include "backend/common/graph_kernel/convert_call_to_prim.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"
#include "symbolic_shape/utils.h"

namespace mindspore::graphkernel::test {
using NodeShapeVector = std::map<AnfNodePtr, ShapeVector>;
class TestKernelPacket : public GraphKernelCommonTestSuite {
 public:
  CNodePtrList GetAllPacketNodes(const FuncGraphPtr &fg);
  bool InferPacketNode(const CNodePtr &packet_node, const NodeShapeVector &input_real_shape);
};
}  // namespace mindspore::graphkernel::test
#endif  // TESTS_UT_CPP_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_COMMON_TEST_SUITE_H_
