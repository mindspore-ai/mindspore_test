/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>
#include <vector>

#include "common/common_test.h"
#include "mindspore/ops/op_def/arithmetic_ops.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "include/common/debug/draw.h"
#include "include/common/utils/convert_utils.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace prim {
class TestGradImplementations : public UT::Common {
 public:
  TestGradImplementations() {}
  virtual void SetUp() {}
};

TEST_F(TestGradImplementations, DISABLED_TestGetAugmentedGraph) {
  FuncGraphPtr fg = ad::g_k_prims.KPrimitive(nullptr, NewValueNode(kPrimScalarMul), nullptr, false);
  ASSERT_TRUE(fg != nullptr);

  auto fg1 = ad::g_k_prims.KPrimitive(nullptr, NewValueNode(kPrimScalarMul), nullptr, false);

  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;
  ASSERT_TRUE(Isomorphic(fg, fg1, &equiv_graph, &equiv_node));
}

}  // namespace prim
}  // namespace mindspore
