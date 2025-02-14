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
#include "ops/op_def.h"
#include "abstract/abstract_value.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/python_adapter.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/cc_implementations.h"

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
  BeginFunc(input_args.size(), "total");
  GenerateFunction();
  return EndFunc();
}

void MetaImpl::set_prim(const PrimitivePtr &prim) { prim_ = prim; }

PrimitivePtr MetaImpl::prim() const { return prim_; }

void MetaImpl::set_check_func(const CheckFunc &check_func) { check_func_ = check_func; }

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

void MetaImpl::BeginFunc(size_t params_size, const std::string &func_name) {
  auto builder = std::make_shared<MetaFuncBuilder>(func_name);
  builder->BeginFunc(params_size);
  func_builder_stack_.push(builder);
}

FuncGraphPtr MetaImpl::EndFunc() {
  if (func_builder_stack_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "func_builder_stack_ is empty when trying to pop MetaFuncBuilder.";
  }
  auto graph = func_builder_stack_.top()->EndFunc();
  func_builder_stack_.pop();
  // Dump IR for top graph.
  if (func_builder_stack_.empty()) {
    DumpIRForMetaDsl(graph);
  }
  return graph;
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

FuncGraphPtr MetaImpl::BuildSubFunction(const std::string &func_name, const BlockFunc &sub_func, size_t n_args) {
  BeginFunc(n_args, func_name);
  sub_func();
  return EndFunc();
}

NodePtr MetaImpl::IfCond(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch,
                         const NodePtrList &args) {
  size_t n_args = args.size();
  auto true_branch_graph = BuildSubFunction("true_branch", true_branch, n_args);
  auto false_branch_graph = BuildSubFunction("false_branch", false_branch, n_args);
  auto bool_condition = NewNode({{NewValueNode(prim::kPrimCond), condition, NewValueNode(MakeValue(false))}});
  auto switch_node = NewNode({NewValueNode(prim::kPrimSwitch), bool_condition, NewValueNode(true_branch_graph),
                              NewValueNode(false_branch_graph)});
  NodePtrList node_list{switch_node};
  (void)std::copy(args.begin(), args.end(), std::back_inserter(node_list));
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

NodePtr MetaImpl::Not(const NodePtr &x) { return NewNode({GetMultitypeOps("logical_not"), x}); }

NodePtr MetaImpl::Raise(const std::string &exception_type, const std::string &exception_msg) {
  auto node = NewNode(
    {NewValueNode(prim::kPrimRaise), NewValueNode(exception_type), NewValueNode(exception_msg), NewValueNode("None")});
  node->func_graph()->set_is_tensor_condition_branch(true);
  return node;
}
}  // namespace mindspore::prim
