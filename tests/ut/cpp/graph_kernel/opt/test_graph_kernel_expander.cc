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

#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_expander_cloud.h"

namespace mindspore::graphkernel::test {
class TestGraphExpanderCheck : public GraphKernelCommonTestSuite {};

namespace {
ValuePtr NewScalar(TypeId type_id, float value = 2.0f) {
  switch (type_id) {
    case kNumberTypeBool:
      return MakeValue(static_cast<bool>(value));
    case kNumberTypeInt8:
      return MakeValue(static_cast<int8_t>(value));
    case kNumberTypeInt16:
      return MakeValue(static_cast<int16_t>(value));
    case kNumberTypeInt32:
      return MakeValue(static_cast<int32_t>(value));
    case kNumberTypeInt64:
      return MakeValue(static_cast<int64_t>(value));
    case kNumberTypeUInt8:
      return MakeValue(static_cast<uint8_t>(value));
    case kNumberTypeUInt16:
      return MakeValue(static_cast<uint16_t>(value));
    case kNumberTypeUInt32:
      return MakeValue(static_cast<uint32_t>(value));
    case kNumberTypeUInt64:
      return MakeValue(static_cast<uint64_t>(value));
    case kNumberTypeFloat32:
      return MakeValue(value);
    case kNumberTypeFloat64:
      return MakeValue(static_cast<double>(value));
    default:
      return nullptr;
  }
}

void RunAddExt(const TypePtr &x0_type, const TypePtr &x1_type, const TypePtr &alpha_type, bool alpha_is_const,
               bool can_expand, GraphKernelCommonTestSuite *t) {
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", x0_type, {32, 32});
  auto x1 = c.NewTensorInput("x1", x1_type, {32, 32});
  CNodePtr op;
  if (alpha_is_const) {
    auto alpha = c.NewValueNode(NewScalar(alpha_type->type_id()));
    op = c.NewCNodeWithBuildInfo("AddExt", {x0, x1, alpha}, {});
  } else {
    ShapeVector shape{};
    auto alpha = c.NewTensorInput("alpha", alpha_type, shape);
    op = c.NewCNodeWithBuildInfo("AddExt", {x0, x1, alpha}, {});
  }
  c.SetOutput(op);
  auto fg = c.GetGraph();

  t->RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = can_expand ? 1 : 0;
  ASSERT_EQ(t->GetAllGKNodes(fg).size(), gk_size);
}
}  // namespace

/// Feature: Test graph kernel expander pass
/// Description: AddExt with different input data types
/// Expectation: AddExt can be expanded only when its input data types are supported.
TEST_F(TestGraphExpanderCheck, add_ext) {
  SetDeviceTarget(kAscendDevice);

  constexpr bool can_expand = true;
  constexpr bool can_not_expand = false;
  RunAddExt(kBool, kBool, kInt64, true, can_not_expand, this);
  RunAddExt(kInt8, kInt8, kInt64, true, can_not_expand, this);
  RunAddExt(kInt16, kInt16, kInt64, true, can_not_expand, this);
  RunAddExt(kInt32, kInt32, kInt64, true, can_expand, this);
  RunAddExt(kInt32, kInt32, kInt64, false, can_not_expand, this);
  RunAddExt(kInt64, kInt64, kInt64, true, can_not_expand, this);
  RunAddExt(kUInt8, kUInt8, kInt64, true, can_not_expand, this);
  RunAddExt(kUInt16, kUInt16, kInt64, true, can_not_expand, this);
  RunAddExt(kUInt32, kUInt32, kInt64, true, can_not_expand, this);
  RunAddExt(kUInt64, kUInt64, kInt64, true, can_not_expand, this);
  RunAddExt(kFloat16, kFloat16, kFloat32, true, can_expand, this);
  RunAddExt(kFloat32, kFloat32, kFloat32, true, can_expand, this);
  RunAddExt(kBFloat16, kBFloat16, kFloat32, true, can_expand, this);
  RunAddExt(kBFloat16, kBFloat16, kInt64, true, can_expand, this);
  RunAddExt(kBFloat16, kBFloat16, kInt64, false, can_not_expand, this);
}
}  // namespace mindspore::graphkernel::test
