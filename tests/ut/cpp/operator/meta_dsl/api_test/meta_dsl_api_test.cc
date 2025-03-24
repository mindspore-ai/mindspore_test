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

#include "ir/manager.h"
#include "ir/graph_utils.h"
#include "common/common_test.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/optimizer.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "tests/ut/cpp/operator/meta_dsl/api_test/api_define.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::prim {
using AbstractTensor = abstract::AbstractTensor;
using AbstractTensorPtr = abstract::AbstractTensorPtr;
using AbstractScalar = abstract::AbstractScalar;
using AbstractScalarPtr = abstract::AbstractScalarPtr;
using AbstractTuple = abstract::AbstractTuple;
using AbstractTuplePtr = abstract::AbstractTuplePtr;
using AbstractList = abstract::AbstractList;
using AbstractListPtr = abstract::AbstractListPtr;

class TestMetaDslApi : public UT::Common {
 public:
  TestMetaDslApi() {}

  FuncGraphPtr NewFuncGraph(const MetaImplPtr &meta, const AbstractBasePtrList &abs_list) {
    // Create FuncGraph.
    FuncGraphPtr fg = std::make_shared<FuncGraph>();
    AnfNodePtrList inputs{NewValueNode(meta)};
    for (size_t i = 0; i < abs_list.size(); ++i) {
      auto param = fg->add_parameter();
      (void)inputs.emplace_back(param);
    }
    CNodePtr cnode = fg->NewCNode(inputs);
    fg->set_output(cnode);
    // Renormalize.
    pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
    return pipeline::Renormalize(resource, fg, abs_list);
  }

  size_t GetPrimitiveSize(const FuncGraphPtr &fg, const PrimitivePtr &prim) {
    auto all_nodes = TopoSort(fg->return_node(), SuccDeeperSimple, AlwaysInclude);
    return std::count_if(all_nodes.begin(), all_nodes.end(),
                         [&](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim); });
  }

  AbstractTensorPtr NewAbstractTensor(int64_t input, const TypePtr &data_type) {
    return std::make_shared<AbstractTensor>(std::make_shared<tensor::Tensor>(input, data_type));
  }

  AbstractTensorPtr NewAbstractTensor(const TypePtr &element_type, const ShapeVector &shape) {
    return std::make_shared<AbstractTensor>(element_type, shape);
  }
};

/// Feature: Meta DSL
/// Description: Test if expression in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_if) {
  auto op = CreateMetaImpl("TestIf");
  AbstractBasePtrList abs_list{NewAbstractTensor(1, kInt32)};
  auto fg = NewFuncGraph(op, abs_list);
  // Check output.
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  MS_EXCEPTION_IF_NULL(out_abs->BuildValue());
  ASSERT_TRUE(out_abs->isa<AbstractScalar>());
  // Check the graph construct.
  ASSERT_GE(GetPrimitiveSize(fg, prim::kPrimSwitch), 1);
}

/// Feature: Meta DSL
/// Description: Test if expression in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_if_exp) {
  auto op = CreateMetaImpl("TestIfExp");
  AbstractBasePtrList abs_list{NewAbstractTensor(1, kInt32), NewAbstractTensor(2, kInt32)};
  auto fg = NewFuncGraph(op, abs_list);
  // Check output.
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  // Check the graph construct.
  ASSERT_EQ(GetPrimitiveSize(fg, prim::kPrimSwitch), 1);
  ASSERT_EQ(GetPrimitiveSize(fg, prim::kPrimAdd), 1);
}

/// Feature: Meta DSL
/// Description: Test for-loop expression in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_for) {
  auto op = CreateMetaImpl("TestFor");
  auto abs_lower = std::make_shared<AbstractScalar>(static_cast<int64_t>(0));
  auto abs_upper = std::make_shared<AbstractScalar>(static_cast<int64_t>(4));
  AbstractBasePtrList abs_list{NewAbstractTensor(0, kInt32), abs_lower, abs_upper};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  ASSERT_EQ(GetPrimitiveSize(fg, prim::kPrimAdd), 1);
}

/// Feature: Meta DSL
/// Description: Test while-loop expression in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_whlie) {
  auto op = CreateMetaImpl("TestWhile");
  AbstractBasePtrList abs_list{NewAbstractTensor(10, kInt32)};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  ASSERT_EQ(GetPrimitiveSize(fg, prim::kPrimAdd), 1);
}

/// Feature: Meta DSL
/// Description: Test Scan in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_scan) {
  auto op = CreateMetaImpl("TestScan");
  AbstractBasePtrList elem_list = {
    NewAbstractTensor(1, kInt32),
    NewAbstractTensor(2, kInt32),
    NewAbstractTensor(3, kInt32),
    NewAbstractTensor(4, kInt32),
  };
  auto abs_tuple = std::make_shared<AbstractTuple>(elem_list);
  AbstractBasePtrList abs_list{NewAbstractTensor(0, kInt32), abs_tuple};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTuple>());
  ASSERT_EQ(GetPrimitiveSize(fg, prim::kPrimScan), 1);
}

/// Feature: Meta DSL
/// Description: Test And in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_and) {
  auto op = CreateMetaImpl("TestAnd");
  auto abs_scalar = std::make_shared<AbstractScalar>(static_cast<int64_t>(0));
  AbstractBasePtrList abs_list{NewAbstractTensor(1, kInt32), abs_scalar};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  ASSERT_GE(GetPrimitiveSize(fg, prim::kPrimSwitch), 1);
}

/// Feature: Meta DSL
/// Description: Test Or in MetaDSL.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_or) {
  auto op = CreateMetaImpl("TestOr");
  auto abs_scalar = std::make_shared<AbstractScalar>(static_cast<int64_t>(0));
  AbstractBasePtrList abs_list{NewAbstractTensor(1, kFloat64), abs_scalar};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  ASSERT_GE(GetPrimitiveSize(fg, prim::kPrimSwitch), 1);
}

/// Feature: Meta DSL
/// Description: Convert TypeId to dtype.
/// Expectation: Run successfully.
TEST_F(TestMetaDslApi, test_dtype) {
  auto op = CreateMetaImpl("TestDtype");
  auto abs_dtype = std::make_shared<AbstractScalar>(static_cast<int64_t>(kNumberTypeInt32));
  AbstractBasePtrList abs_list{NewAbstractTensor(1, kFloat64), NewAbstractTensor(2, kFloat64), abs_dtype};
  auto fg = NewFuncGraph(op, abs_list);
  auto out_abs = fg->return_node()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  ASSERT_TRUE(out_abs->isa<AbstractTensor>());
  auto type_id = out_abs->cast<AbstractTensorPtr>()->element()->BuildType()->type_id();
  ASSERT_TRUE(type_id == kNumberTypeInt32);
}
}  // namespace mindspore::prim
