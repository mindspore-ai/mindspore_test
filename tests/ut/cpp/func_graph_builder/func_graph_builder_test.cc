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

#include <string>
#include "ir/tensor.h"
#include "frontend/ir/tensor_py.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "pipeline/jit/pi/graph_capture/abstract_wrapper.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils.h"
#include "mindspore/ops/op_def/arithmetic_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace pijit {
class TestFuncGraphBuilder : public UT::Common {
 public:
  TestFuncGraphBuilder() : get_py_fun_("gtest_input.pipeline.pi.func_graph_builder", true) {}

  bool CheckEqual(const FuncGraphPtr &fg1, const FuncGraphPtr &fg2) {
    equiv_graph_.clear();
    equiv_node_.clear();
    return Isomorphic(fg1, fg2, &equiv_graph_, &equiv_node_);
  }

  FuncGraphBuilderPtr CreateFuncGraphBuilder() { return std::make_shared<FuncGraphBuilder>(true); }

  AbstractBasePtr CreateAbstractTensor(TypeId type_id, const ShapeVector &shape) {
    return std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape);
  }

  AbstractBasePtr CreateAbstractRefTensor(const abstract::AbstractTensorPtr &ref_value, const ValuePtr &ref_key_value) {
    return std::make_shared<abstract::AbstractRefTensor>(ref_value, ref_key_value);
  }

  AbstractBasePtr CreateAbstractScalar(const std::string &str) {
    return std::make_shared<abstract::AbstractScalar>(str);
  }

  AbstractBasePtr CreateAbstractScalar(int64_t value) { return std::make_shared<abstract::AbstractScalar>(value); }

  AbstractBasePtr CreateAbstractTuple(const AbstractBasePtrList &elements) {
    return std::make_shared<abstract::AbstractTuple>(elements);
  }

  AbstractBasePtr CreateAbstractList(const AbstractBasePtrList &elements) {
    return std::make_shared<abstract::AbstractList>(elements);
  }

  AbstractBasePtr CreateAbstractDict(const std::vector<abstract::AbstractElementPair> &elements) {
    return std::make_shared<abstract::AbstractDictionary>(elements);
  }

  py::object CreateIntTensorObject(const ShapeVector &shape_vec) {
    py::tuple shape(shape_vec.size());
    for (size_t i = 0; i < shape_vec.size(); ++i) {
      shape[i] = py::int_(shape_vec[i]);
    }
    constexpr auto func_graph_builder_mod = "gtest_input.pipeline.pi.func_graph_builder";
    py::module mod = python_adapter::GetPyModule(func_graph_builder_mod);
    return python_adapter::CallPyModFn(mod, "create_int_tensor", shape);
  }

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
  FuncGraphPairMapEquiv equiv_graph_;
  NodeMapEquiv equiv_node_;
};

class TestAddLocalVariable : public TestFuncGraphBuilder {
 public:
  py::object GetParameterObj() {
    constexpr auto common_module = "mindspore.common";
    py::module mod = python_adapter::GetPyModule(common_module);
    auto py_tensor = python_adapter::CallPyModFn(mod, "Tensor", 1.1);
    auto py_parameter = python_adapter::CallPyModFn(mod, "Parameter", py_tensor);
    return py_parameter;
  }
};

// Feature: Build graph in pi_jit.
// Description: Test function AddTopGraphArgInput.
// Expectation: The result wrapper is correct.
TEST_F(TestFuncGraphBuilder, TestAddTopGraphArgInput) {
  auto builder = CreateFuncGraphBuilder();
  const py::object &input1 = py::int_(1);
  const auto &v1_wrapper = builder->AddTopGraphArgInput(input1);
  const auto &expect_v1_abstract = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(1));
  ASSERT_TRUE(*(v1_wrapper->abstract()) == *expect_v1_abstract);
  const auto &input2 = CreateIntTensorObject(ShapeVector{2, 3});
  const auto &v2_wrapper = builder->AddTopGraphArgInput(input2);
  const auto &expect_v2_abstract = CreateAbstractTensor(kNumberTypeInt64, ShapeVector{2, 3});
  ASSERT_TRUE(*(v2_wrapper->abstract()) == *expect_v2_abstract);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode.
// Expectation: The expected node is constructed.
TEST_F(TestFuncGraphBuilder, TestAddNodeAndAddOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();

  auto obj = func_graph_builder.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  const auto &expect_abstract = CreateAbstractScalar(3);
  ASSERT_EQ(*(obj->abstract()), *expect_abstract);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_EQ(graph, nullptr);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddNodeAndMultiOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node", "graph_multi_output");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with constant input.
// Expectation: Failed to add the node.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddNodeConstantInput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto obj = func_graph_builder.AddNode(prim::kPrimScalarAdd, {input1, v2_wrapper});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node_with_constant", "graph");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with an uncallable object.
// Expectation: Failed to add the node.
TEST_F(TestFuncGraphBuilder, TestAddNodeUnCallable) {
  FuncGraphBuilder func_graph_builder(true);
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto obj = func_graph_builder.AddNode(scalar_add_prim_class, {input1, input2});
  ASSERT_EQ(obj, nullptr);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with constant input.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddMultiNode) {
  FuncGraphBuilder func_graph_builder(true);
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto add_wrapper = func_graph_builder.AddMultiNode("add", {input1, input2});
  ASSERT_NE(add_wrapper, nullptr);
  auto add_abstract = add_wrapper->abstract();
  ASSERT_NE(add_abstract, nullptr);
  auto expected_add_abstract = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(3));
  ASSERT_EQ(*add_abstract == *expected_add_abstract, true);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add func_graph called node.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddFgCallNodeSingleOutput) {
  FuncGraphBuilder func_graph_builder1(true);
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder1.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder1.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder1.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder1.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj));
  auto graph1 = func_graph_builder1.graph();
  ASSERT_NE(graph1, nullptr);

  FuncGraphBuilder func_graph_builder2(true);
  v1_wrapper = func_graph_builder2.AddLocalVariable(int_v1);
  input1 = func_graph_builder2.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  v2_wrapper = func_graph_builder2.AddLocalVariable(int_v2);
  input2 = func_graph_builder2.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto call_node_obj = func_graph_builder2.AddNode(graph1, {input1, input2});
  ASSERT_NE(call_node_obj, nullptr);
  ASSERT_TRUE(func_graph_builder2.AddOutput(call_node_obj));
  auto graph2 = func_graph_builder2.graph();
  DumpIR("graph2.ir", graph2);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_fg_call_node", "graph_single_output");
  ASSERT_TRUE(CheckEqual(graph2, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add func_graph called node.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddFgCallNodeMultiOutput) {
  FuncGraphBuilder func_graph_builder1(true);
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder1.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder1.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder1.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder1.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj1 = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj1, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj1));
  auto obj2 = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj2, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj2));
  auto graph1 = func_graph_builder1.graph();
  ASSERT_NE(graph1, nullptr);

  FuncGraphBuilder func_graph_builder2(true);
  v1_wrapper = func_graph_builder2.AddLocalVariable(int_v1);
  input1 = func_graph_builder2.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  v2_wrapper = func_graph_builder2.AddLocalVariable(int_v2);
  input2 = func_graph_builder2.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto call_node_obj = func_graph_builder2.AddNode(graph1, {input1, input2});
  ASSERT_NE(call_node_obj, nullptr);
  ASSERT_TRUE(func_graph_builder2.AddOutput(call_node_obj));
  auto graph2 = func_graph_builder2.graph();
  DumpIR("graph2.ir", graph2);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_fg_call_node", "graph_multi_output");
  ASSERT_TRUE(CheckEqual(graph2, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to get the function or primitive from a method.
// Expectation: Get the correct function or primitive.
TEST_F(TestFuncGraphBuilder, DISABLED_TestGetFunctionFromMethod) {
  py::tuple t;
  auto func = FuncGraphBuilder::ConvertMethod(t.attr("index"));
  ASSERT_NE(func.ptr(), nullptr);
  ASSERT_EQ(func.attr("__name__").cast<std::string>(), "sequence_index");

  func = FuncGraphBuilder::ConvertMethod(t.attr("__getitem__"));
  ASSERT_NE(func.ptr(), nullptr);
  ASSERT_EQ(func.attr("name").cast<std::string>(), prim::kPrimTupleGetItem->name());
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to get the callable obj from a function.
// Expectation: Get the correct callable obj.
TEST_F(TestFuncGraphBuilder, TestGetCallableObjFromFunction) {
  auto operator_mod = python_adapter::GetPyModule("operator");
  auto func_add = python_adapter::GetPyObjAttr(operator_mod, "add");
  auto callable_obj = FuncGraphBuilder::ConvertFunction(func_add);
  ASSERT_NE(callable_obj.ptr(), nullptr);
  ASSERT_TRUE(py::isinstance<MetaFuncGraph>(callable_obj));

  auto builtin_mod = python_adapter::GetPyModule("builtins");
  auto func_abs = python_adapter::GetPyObjAttr(builtin_mod, "abs");
  callable_obj = FuncGraphBuilder::ConvertFunction(func_abs);
  ASSERT_NE(callable_obj.ptr(), nullptr);
  ASSERT_EQ(callable_obj.attr("name").cast<std::string>(), prim::kPrimInnerAbs->name());
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to check if an obj can be constantly folded.
// Expectation: Get the correct result.
TEST_F(TestFuncGraphBuilder, TestCanConstantFoldFunc) {
  auto operator_mod = python_adapter::GetPyModule("operator");
  auto func_add = python_adapter::GetPyObjAttr(operator_mod, "add");
  ASSERT_TRUE(FuncGraphBuilder::CanConstantFoldFunc(func_add));

  auto builtin_mod = python_adapter::GetPyModule("builtins");
  auto func_abs = python_adapter::GetPyObjAttr(builtin_mod, "abs");
  ASSERT_TRUE(FuncGraphBuilder::CanConstantFoldFunc(func_abs));

  auto ms_mod = python_adapter::GetPyModule("mindspore");
  auto func_ms_memory_recycle = python_adapter::GetPyObjAttr(builtin_mod, "ms_memory_recycle");
  ASSERT_FALSE(FuncGraphBuilder::CanConstantFoldFunc(func_ms_memory_recycle));
}

// Feature: Add local variable into func_graph_builder
// Description: Test the input which doesn't contain Parameter.
// Expectation: The expected abstract_wrapper is constructed.
TEST_F(TestAddLocalVariable, NotContainsParameter) {
  FuncGraphBuilder func_graph_builder;

  py::int_ m_int = 1;
  auto wrapper = func_graph_builder.AddLocalVariable(m_int);
  ASSERT_NE(wrapper, nullptr);
  const auto &expect_abstract_int = CreateAbstractScalar(1);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_int);

  py::tuple m_tuple = py::make_tuple(1, 2, 3);
  wrapper = func_graph_builder.AddLocalVariable(m_tuple);
  ASSERT_NE(wrapper, nullptr);
  std::vector<AbstractBasePtr> m_vector = {CreateAbstractScalar(1), CreateAbstractScalar(2), CreateAbstractScalar(3)};
  const auto &expect_abstract_tuple = CreateAbstractTuple(m_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_tuple);

  py::list m_list = py::cast(std::vector<int>{1, 2, 3});
  wrapper = func_graph_builder.AddLocalVariable(m_list);
  ASSERT_NE(wrapper, nullptr);
  const auto &expect_abstract_list = CreateAbstractList(m_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_list);

  py::dict m_dict = py::dict();
  m_dict[py::str("key1")] = 1;
  m_dict[py::str("key2")] = 2;
  wrapper = func_graph_builder.AddLocalVariable(m_dict);
  ASSERT_NE(wrapper, nullptr);
  std::vector<abstract::AbstractElementPair> m_pair_vector = {
    std::make_pair(CreateAbstractScalar("key1"), CreateAbstractScalar(1)),
    std::make_pair(CreateAbstractScalar("key2"), CreateAbstractScalar(2)),
  };
  const auto &expect_abstract_dict = CreateAbstractDict(m_pair_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_dict);
}

// Feature: Add local variable into func_graph_builder.
// Description: Test the input which is Parameter.
// Expectation: The expected abstract_wrapper is constructed.
TEST_F(TestAddLocalVariable, IsParameterObject) {
  FuncGraphBuilder func_graph_builder;
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  parse::Parser::UpdateTopFuncGraph(graph);
  auto py_parameter = GetParameterObj();

  auto wrapper = func_graph_builder.AddLocalVariable(py_parameter);
  ASSERT_NE(wrapper, nullptr);
  const auto &abstract_tensor = CreateAbstractTensor(kNumberTypeFloat32, ShapeVector{});
  const auto &expect_abstract_ref_tensor =
    CreateAbstractRefTensor(std::dynamic_pointer_cast<abstract::AbstractTensor>(abstract_tensor), kValueAny);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_ref_tensor);
}

// Feature: Add local variable into func_graph_builder.
// Description: Test the input which is Parameter sequence.
// Expectation: The expected abstract_wrapper is constructed.
TEST_F(TestAddLocalVariable, IsParameterSequence) {
  FuncGraphBuilder func_graph_builder;
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  parse::Parser::UpdateTopFuncGraph(graph);
  auto py_parameter = GetParameterObj();

  py::tuple m_tuple = py::make_tuple(py_parameter);
  auto wrapper = func_graph_builder.AddLocalVariable(m_tuple);
  ASSERT_NE(wrapper, nullptr);
  const auto &abstract_tensor = CreateAbstractTensor(kNumberTypeFloat32, ShapeVector{});
  std::vector<AbstractBasePtr> m_vector = {
    CreateAbstractRefTensor(std::dynamic_pointer_cast<abstract::AbstractTensor>(abstract_tensor), kValueAny)};
  const auto &expect_abstract_tuple = CreateAbstractTuple(m_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_tuple);

  py::list m_list = py::list();
  m_list.append(py_parameter);
  wrapper = func_graph_builder.AddLocalVariable(m_list);
  ASSERT_NE(wrapper, nullptr);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_tuple);
}

// Feature: Add local variable into func_graph_builder.
// Description: Test the input which is not Parameter sequence.
// Expectation: The expected abstract_wrapper is constructed.
TEST_F(TestAddLocalVariable, IsNotParameterSequence) {
  FuncGraphBuilder func_graph_builder;
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  parse::Parser::UpdateTopFuncGraph(graph);
  auto py_parameter = GetParameterObj();

  py::tuple m_tuple = py::make_tuple(1, py_parameter);
  auto wrapper = func_graph_builder.AddLocalVariable(m_tuple);
  ASSERT_NE(wrapper, nullptr);
  const auto &abstract_tensor = CreateAbstractTensor(kNumberTypeFloat32, ShapeVector{});
  const auto &abstract_ref_tensor =
    CreateAbstractRefTensor(std::dynamic_pointer_cast<abstract::AbstractTensor>(abstract_tensor), kValueAny);
  std::vector<AbstractBasePtr> m_vector = {CreateAbstractScalar(1), abstract_ref_tensor};
  const auto &expect_abstract_tuple = CreateAbstractTuple(m_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_tuple);

  py::list m_list = py::list();
  m_list.append(1);
  m_list.append(py_parameter);
  wrapper = func_graph_builder.AddLocalVariable(m_list);
  ASSERT_NE(wrapper, nullptr);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_tuple);

  py::dict m_dict = py::dict();
  m_dict[py::str("key1")] = 1;
  m_dict[py::str("key2")] = py_parameter;
  wrapper = func_graph_builder.AddLocalVariable(m_dict);
  ASSERT_NE(wrapper, nullptr);
  std::vector<abstract::AbstractElementPair> m_pair_vector = {
    std::make_pair(CreateAbstractScalar("key1"), CreateAbstractScalar(1)),
    std::make_pair(CreateAbstractScalar("key2"), abstract_ref_tensor)};
  const auto &expect_abstract_dict = CreateAbstractDict(m_pair_vector);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_dict);

  py::dict m_dict2 = py::dict();
  m_dict2[py::str("key1")] = py_parameter;
  wrapper = func_graph_builder.AddLocalVariable(m_dict2);
  ASSERT_NE(wrapper, nullptr);
  std::vector<abstract::AbstractElementPair> m_pair_vector2 = {
    std::make_pair(CreateAbstractScalar("key1"), abstract_ref_tensor)};
  const auto &expect_abstract_dict2 = CreateAbstractDict(m_pair_vector2);
  ASSERT_EQ(*(wrapper->abstract()), *expect_abstract_dict2);
}

// Feature: Add local variable into func_graph_builder.
// Description: Test the input which is NullPyObj.
// Expectation: The expected abstract_wrapper is constructed.
TEST_F(TestAddLocalVariable, NullPyObj) {
  FuncGraphBuilder func_graph_builder;

  auto wrapper = func_graph_builder.AddLocalVariable(py::buffer());
  ASSERT_EQ(wrapper, nullptr);
}
}  // namespace pijit
}  // namespace mindspore
