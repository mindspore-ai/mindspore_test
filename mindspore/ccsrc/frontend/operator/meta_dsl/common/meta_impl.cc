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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"
#include <algorithm>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ops/op_def.h"
#include "abstract/abstract_value.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/python_adapter.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/cc_implementations.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/parse/parse_base.h"

namespace mindspore::prim {
namespace {
std::unordered_map<std::string, CreateFunc> &GetMetaImplTable() {
  static std::unordered_map<std::string, CreateFunc> meta_impl_table;
  return meta_impl_table;
}

ValuePtr GetPyValueWithCache(const std::string &module_name, const std::string &op_name) {
  static std::map<std::string, ValuePtr> py_value_map;
  auto full_name = module_name + "." + op_name;
  auto iter = py_value_map.find(full_name);
  if (iter != py_value_map.end()) {
    return iter->second;
  }
  auto value = prim::GetPythonOps(op_name, module_name);
  MS_EXCEPTION_IF_NULL(value);
  py_value_map[full_name] = value;
  return value;
}

ValueNodePtr GetMultitypeOps(const std::string &op_name) {
  return NewValueNode(GetPyValueWithCache("mindspore.ops.composite.multitype_ops", op_name));
}

ValuePtr CreateCalssType(const std::string &module_name, const std::string &op_name) {
  py::object obj = python_adapter::GetPyFn(module_name, op_name);
  return std::make_shared<parse::ClassType>(obj, op_name);
}

ValuePtr GetClassTypeValue(const TypeId &type) {
  static std::map<TypeId, ValuePtr> class_type_map = {
    {TypeId::kObjectTypeTensorType, CreateCalssType("mindspore.common.tensor", "Tensor")},
    {TypeId::kNumberTypeInt, CreateCalssType("builtins", "int")},
    {TypeId::kNumberTypeFloat, CreateCalssType("builtins", "float")},
    {TypeId::kNumberTypeBool, CreateCalssType("builtins", "bool")},
    {TypeId::kObjectTypeTuple, CreateCalssType("builtins", "tuple")},
    {TypeId::kObjectTypeList, CreateCalssType("builtins", "list")},
    {TypeId::kObjectTypeString, CreateCalssType("builtins", "str")},
    {TypeId::kNumberTypeComplex, CreateCalssType("builtins", "complex")},
    {TypeId::kObjectTypeNumber, CreateCalssType("numbers", "Number")}};
  const auto &iter = class_type_map.find(type);
  if (iter == class_type_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported TypeId '" << TypeIdToString(type) << "'.";
  }
  return iter->second;
}
}  // namespace

bool IsMetaImpl(const std::string &name) {
  const auto &meta_impl_table = GetMetaImplTable();
  return meta_impl_table.find(name) != meta_impl_table.end();
}

void AddMetaImpl(const std::string &name, const CreateFunc &creator) {
  (void)GetMetaImplTable().emplace(name, creator);
}

MetaImplPtr CreateMetaImpl(const std::string &name) {
  auto &creators = GetMetaImplTable();
  auto it = creators.find(name);
  if (it == creators.end()) {
    MS_LOG(EXCEPTION) << "Failed to create MetaImpl: " << name;
  }
  return it->second();
}

FuncGraphPtr MetaImpl::GenerateFuncGraph(const AbstractBasePtrList &input_args) {
  CheckInputs(input_args);
  BeginFunc("total");
  GenerateFunction();
  return EndFunc();
}

void MetaImpl::set_prim(const PrimitivePtr &prim) { prim_ = prim; }

PrimitivePtr MetaImpl::prim() const { return prim_; }

void MetaImpl::set_check_func(const CheckFunc &check_func) { check_func_ = check_func; }

void MetaImpl::set_bprop_func(const std::function<std::shared_ptr<MetaImpl>()> &bprop_func) {
  bprop_func_ = bprop_func;
}

void MetaImpl::CheckInputs(const AbstractBasePtrList &input_args) const {
  // Check inputs' number.
  const auto &op_def = ops::GetOpDef(name_);
  if (op_def != nullptr) {
    auto args_size = op_def->args_.size();
    if (input_args.size() != args_size) {
      MS_LOG(EXCEPTION) << name_ << " requires " << args_size << " arguments, but got " << input_args.size() << ".";
    }
  }
  // Check inputs' abstract.
  if (check_func_ != nullptr && prim_ != nullptr) {
    check_func_(prim_, input_args);
  }
}

void MetaImpl::BeginFunc(const std::string &func_name) {
  auto builder = std::make_shared<MetaFuncBuilder>(name_ + "_" + func_name);
  builder->BeginFunc();
  func_builder_stack_.push(builder);
}

FuncGraphPtr MetaImpl::EndFunc() {
  if (func_builder_stack_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "func_builder_stack_ is empty when trying to pop MetaFuncBuilder.";
  }
  auto graph = func_builder_stack_.top()->EndFunc();
  func_builder_stack_.pop();
  // Process top graph.
  if (func_builder_stack_.empty()) {
    // Define custom bprop for current op.
    DefineCustomBprop(graph);
    // Dump IR for top graph.
    DumpIRForMetaDsl(graph);
  }
  return graph;
}

void MetaImpl::DefineCustomBprop(const FuncGraphPtr &graph) {
  if (bprop_func_ != nullptr) {
    auto bprop_meta_impl = bprop_func_();
    // Create bprop graph.
    bprop_graph_ = std::make_shared<FuncGraph>();
    bprop_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
    MS_EXCEPTION_IF_NULL(bprop_graph_->debug_info());
    bprop_graph_->debug_info()->set_name(name_ + "_" + parse::CUSTOM_BPROP_NAME);
    // Implement bprop graph.
    constexpr auto extend_size = 2;
    auto params_size = graph->parameters().size() + extend_size;
    AnfNodePtrList inputs{NewValueNode(bprop_meta_impl)};
    for (size_t i = 0; i < params_size; ++i) {
      (void)inputs.emplace_back(bprop_graph_->add_parameter());
    }
    CNodePtr cnode = bprop_graph_->NewCNodeInOrder(inputs);
    bprop_graph_->set_output(cnode);
    // Associate bprop to graph.
    MS_LOG(DEBUG) << "Define custom bprop for " << name_ << ": " << bprop_graph_->ToString();
    (void)graph->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph_));
    (void)bprop_graph_->transforms().emplace("primal", FuncGraphTransform(graph));
    graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
  }
}

void MetaImpl::DumpIRForMetaDsl(const FuncGraphPtr &graph) const {
  std::string str(common::GetCompileConfig("DUMP_IR_META_DSL"));
  if (!str.empty() && str == name_) {
    std::string file_name = "MetaDSL_" + name_ + ".ir";
    DumpIR(file_name, graph);
  }
}

void MetaImpl::Return(const NodePtr &output) { func_builder_stack_.top()->SetOutput(output); }

NodePtr MetaImpl::NewParam(const std::string &name) { return func_builder_stack_.top()->AddParameter(name); }

NodePtr MetaImpl::NewNode(const NodePtrList &nodes) { return func_builder_stack_.top()->CreateNode(nodes); }

FuncGraphPtr MetaImpl::BuildSubFunction(const std::string &func_name, const BlockFunc &sub_func) {
  BeginFunc(func_name);
  sub_func();
  return EndFunc();
}

NodePtr MetaImpl::IfCond(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch,
                         const NodePtrList &args) {
  auto true_branch_graph = BuildSubFunction("true_branch", true_branch);
  auto false_branch_graph = BuildSubFunction("false_branch", false_branch);
  auto bool_condition = NewNode({{NewValueNode(prim::kPrimCond), condition, NewValueNode(MakeValue(false))}});
  auto switch_node = NewNode({NewValueNode(prim::kPrimSwitch), bool_condition, NewValueNode(true_branch_graph),
                              NewValueNode(false_branch_graph)});
  NodePtrList node_list{switch_node};
  (void)std::copy(args.begin(), args.end(), std::back_inserter(node_list));
  return NewNode(node_list);
}

NodePtr MetaImpl::For(const NodePtr &lower, const NodePtr &upper,
                      const std::function<void(const NodePtr &, const NodePtr &)> &loop_func, const NodePtr &init_val) {
  // Build graph for loop body.
  BeginFunc("for_loop");
  auto param_index = NewParam("index");
  auto param_value = NewParam("value");
  loop_func(param_index, param_value);
  auto loop_graph = EndFunc();
  // Create node with ForiLoop.
  NodePtrList node_list{NewValueNode(prim::kPrimForiLoop), lower, upper, NewValueNode(loop_graph), init_val};
  return NewNode(node_list);
}

NodePtr MetaImpl::While(const std::function<void(const NodePtr &)> &cond_func,
                        const std::function<void(const NodePtr &)> &loop_func, const NodePtr &init_val) {
  // Build graph for condition.
  BeginFunc("while_cond");
  auto param_cond = NewParam("cond");
  cond_func(param_cond);
  auto cond_graph = EndFunc();
  // Build graph for loop body.
  BeginFunc("while_loop");
  auto param_loop = NewParam("loop");
  loop_func(param_loop);
  auto loop_graph = EndFunc();
  // Create node with WhileLoop.
  NodePtrList node_list{NewValueNode(prim::kPrimWhileLoop), NewValueNode(cond_graph), NewValueNode(loop_graph),
                        init_val};
  return NewNode(node_list);
}

NodePtr MetaImpl::Scan(const std::function<void(const NodePtr &, const NodePtr &)> &loop_func, const NodePtr &init,
                       const NodePtr &xs, const NodePtr &length) {
  // Build graph for loop body.
  BeginFunc("scan_loop");
  auto param_result = NewParam("result");
  auto param_elem = NewParam("elem");
  loop_func(param_result, param_elem);
  auto loop_graph = EndFunc();
  // Create node with Scan.
  NodePtrList node_list{NewValueNode(prim::kPrimScan), NewValueNode(loop_graph), init, xs, length};
  return NewNode(node_list);
}

NodePtr MetaImpl::Equal(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("equal"), x, y}); }

NodePtr MetaImpl::NotEqual(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("not_equal"), x, y}); }

NodePtr MetaImpl::Greater(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("greater"), x, y}); }

NodePtr MetaImpl::Less(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("less"), x, y}); }

NodePtr MetaImpl::GreaterEqual(const NodePtr &x, const NodePtr &y) {
  return NewNode({GetMultitypeOps("greater_equal"), x, y});
}

NodePtr MetaImpl::LessEqual(const NodePtr &x, const NodePtr &y) {
  return NewNode({GetMultitypeOps("less_equal"), x, y});
}

NodePtr MetaImpl::GetItem(const NodePtr &x, const NodePtr &y) { return NewNode({GetMultitypeOps("getitem"), x, y}); }

NodePtr MetaImpl::SetItem(const NodePtr &x, const NodePtr &y, const NodePtr &z) {
  return NewNode({GetMultitypeOps("setitem"), x, y, z});
}

NodePtr MetaImpl::IsNone(const NodePtr &node) {
  return NewNode({NewValueNode(prim::kPrimIs_), node, NewValueNode(kNone)});
}

NodePtr MetaImpl::IsNotNone(const NodePtr &node) {
  return NewNode({NewValueNode(prim::kPrimIsNot), node, NewValueNode(kNone)});
}

NodePtr MetaImpl::And(const NodePtr &x, const NodePtr &y) {
  auto true_branch = [&]() { Return(y); };
  auto false_branch = [&]() { Return(x); };
  return IfCond(x, true_branch, false_branch, {x, y});
}

NodePtr MetaImpl::Or(const NodePtr &x, const NodePtr &y) {
  auto true_branch = [&]() { Return(x); };
  auto false_branch = [&]() { Return(y); };
  return IfCond(x, true_branch, false_branch, {x, y});
}

NodePtr MetaImpl::ScalarAdd(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarAdd), x, y});
}

NodePtr MetaImpl::ScalarSub(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarSub), x, y});
}

NodePtr MetaImpl::ScalarMul(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarMul), x, y});
}

NodePtr MetaImpl::ScalarFloorDiv(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarFloorDiv), x, y});
}

NodePtr MetaImpl::ScalarMod(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarMod), x, y});
}

NodePtr MetaImpl::ScalarPow(const NodePtr &x, const NodePtr &y) {
  return NewNode({NewValueNode(prim::kPrimScalarPow), x, y});
}

NodePtr MetaImpl::Shape(const NodePtr &x) { return NewNode({NewValueNode(prim::kPrimShape), x}); }

NodePtr MetaImpl::Rank(const NodePtr &x) { return NewNode({NewValueNode(prim::kPrimRank), x}); }

NodePtr MetaImpl::Reshape(const NodePtr &x, const NodePtr &shape) {
  return NewNode({NewValueNode(prim::kPrimReshape), x, shape});
}

NodePtr MetaImpl::Not(const NodePtr &x) { return NewNode({GetMultitypeOps("logical_not"), x}); }

NodePtr MetaImpl::Raise(const std::string &exception_type, const std::string &exception_msg) {
  auto node = NewNode(
    {NewValueNode(prim::kPrimRaise), NewValueNode(exception_type), NewValueNode(exception_msg), NewValueNode("None")});
  node->func_graph()->set_is_tensor_condition_branch(true);
  return node;
}

NodePtr MetaImpl::IsInstance(const NodePtr &x, const TypeId &type) {
  auto class_type = NewValueNode(GetClassTypeValue(type));
  return NewNode({NewValueNode(prim::kPrimIsInstance), x, class_type});
}

NodePtr MetaImpl::IsInstance(const NodePtr &x, const std::vector<TypeId> &types) {
  NodePtrList class_type_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::transform(types.begin(), types.end(), std::back_inserter(class_type_list),
                       [](const auto &type) { return NewValueNode(GetClassTypeValue(type)); });
  auto class_type_list_node = NewNode(class_type_list);
  return NewNode({NewValueNode(prim::kPrimIsInstance), x, class_type_list_node});
}
}  // namespace mindspore::prim
