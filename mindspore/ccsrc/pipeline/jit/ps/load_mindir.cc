/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/ps/load_mindir.h"

#include <string>
#include <set>
#include <memory>
#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>

#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops/op_def.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "utils/check_convert_utils.h"
#include "load_mindir/infer_mindir.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/sequence_ops.h"

namespace mindspore {
namespace pipeline {
bool InferMindIR(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &root = resource->func_graph();
  InferFuncGraphLoaded(root);
  return true;
}

std::vector<AnfNodePtr> ArgsNeededToConvert(const PrimitivePtr &prim, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  std::vector<AnfNodePtr> prim_init_arg_nodes;
  MS_EXCEPTION_IF_NULL(op_def);
  // Get init args.
  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      auto arg_name = op_arg.arg_name_;
      ValuePtr attr;
      // "data_format" is renamed as "format" for some operator.
      if (CheckAndConvertUtils::CheckPrimAttrConverted(prim->name()) && arg_name == "data_format" &&
          prim->HasAttr("format")) {
        attr = prim->GetAttr("format");
      } else if (!prim->HasAttr(arg_name)) {
        attr = parse::GetArgDefaultValue(prim->name(), arg_name);
        if (attr == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot find attribute: " << arg_name << " from primitive :" << prim->name();
        }
      } else {
        attr = prim->GetAttr(arg_name);
      }
      (void)prim_init_arg_nodes.emplace_back(NewValueNode(attr));
    }
  }
  return prim_init_arg_nodes;
}

void ModifyOneCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (IsValueNode<Primitive>(inputs[0])) {
    auto prim = GetValueNode<PrimitivePtr>(inputs[0]);
    if (mindspore::ops::IsPrimitiveFunction(prim->name())) {
      // Append Primitive arguments to the inputs.
      std::vector<AnfNodePtr> prim_init_arg_nodes = ArgsNeededToConvert(prim, cnode);
      // Get call args.
      AnfNodePtrList prim_call_arg_nodes(inputs.begin() + 1, inputs.end());
      // Create new node.
      auto new_prim = std::make_shared<Primitive>(*prim);
      AnfNodePtrList input_nodes{NewValueNode(new_prim)};
      (void)std::copy(prim_call_arg_nodes.cbegin(), prim_call_arg_nodes.cend(), std::back_inserter(input_nodes));
      (void)std::copy(prim_init_arg_nodes.cbegin(), prim_init_arg_nodes.cend(), std::back_inserter(input_nodes));
      auto new_cnode = func_graph->NewCNodeInOrder(input_nodes);
      MS_LOG(DEBUG) << "Convert primitive args: " << prim->name() << ". node: " << cnode->DebugString()
                    << ", new_node: " << new_cnode->DebugString();
      auto manager = func_graph->manager();
      if (manager == nullptr) {
        manager = MakeManager();
        manager->AddFuncGraph(func_graph, true);
      }
      (void)manager->Replace(cnode, new_cnode);
    }
  }
}

void ModifyOneFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *func_graph_set,
                        std::set<FuncGraphPtr> *func_graph_modified) {
  MS_LOG(DEBUG) << "Start modifying: " << func_graph->ToString();
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    ModifyOneCNode(func_graph, cnode);
    auto &inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsValueNode<FuncGraph>(inputs[i])) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(inputs[i]);
        if ((*func_graph_set).find(fg) == (*func_graph_set).end() &&
            (*func_graph_modified).find(fg) == (*func_graph_modified).end()) {
          (void)(*func_graph_set).insert(fg);
        }
      }
    }
  }
}

void ModifyGraphs(const FuncGraphPtr &func_graph) {
  std::set<FuncGraphPtr> func_graph_set{};
  std::set<FuncGraphPtr> func_graph_modified{};
  (void)func_graph_set.insert(func_graph);
  // Check every node in every graph to find nodes needed to convert.
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.cbegin();
    if (!func_graph->has_flag("generated_from_mindir_with_prim_func")) {
      ModifyOneFuncGraph(fg, &func_graph_set, &func_graph_modified);
    }
    (void)func_graph_set.erase(fg);
    (void)func_graph_modified.insert(fg);
  }
}

bool ModifyGraphGeneratedByMindIR(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  ModifyGraphs(func_graph);
  return true;
}

namespace {

ValueNodePtr NewDtypeValueNode(mindspore::TypeId dtype) {
  auto dtype_value = MakeValue(static_cast<int64_t>(dtype));
  auto dtype_node = NewValueNode(dtype_value);
  dtype_node->set_abstract(dtype_value->ToAbstract());
  return dtype_node;
}

AnfNodePtr ConvertScalarDtype(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const TypePtr &from_type,
                              const TypePtr &to_type, const std::function<ValuePtr(const ValuePtr &)> &convert_func) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_scalar = abs->cast<abstract::AbstractScalarPtr>();
  if (abs_scalar == nullptr) {
    MS_LOG(INFO) << "It should be an AbstractScalar, but is: " << abs->ToString();
    return nullptr;
  }
  TypePtr tp = abs_scalar->GetType();
  if (tp != from_type) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "Convert scalar dtype from " << from_type->ToString() << " to " << to_type->ToString()
                << ", node: " << node->DebugString() << ", abstract: " << abs->ToString();
  auto scalar_value = abs_scalar->GetValue();
  MS_EXCEPTION_IF_NULL(scalar_value);

  if (scalar_value->ContainsValueAny()) {
    ValueNodePtr dtype_node = NewDtypeValueNode(to_type->type_id());
    auto cast_node = func_graph->NewCNode({NewValueNode(prim::kPrimScalarCast), node, dtype_node});
    auto new_abs = std::make_shared<abstract::AbstractScalar>(*abs_scalar);
    new_abs->set_type(to_type);
    cast_node->set_abstract(new_abs);
    return cast_node;
  } else {
    auto new_value = convert_func(scalar_value);
    auto new_node = NewValueNode(new_value);
    new_node->set_abstract(new_value->ToAbstract());
    return new_node;
  }
}

ValuePtr ConvertFp32ValueToFp64(const ValuePtr &scalar_value) {
  auto fp32_value = scalar_value->cast<FP32ImmPtr>();
  MS_EXCEPTION_IF_NULL(fp32_value);
  return MakeValue(static_cast<double>(fp32_value->value()));
}

ValuePtr ConvertInt32ValueToInt64(const ValuePtr &scalar_value) {
  auto int32_value = scalar_value->cast<Int32ImmPtr>();
  MS_EXCEPTION_IF_NULL(int32_value);
  return MakeValue(static_cast<int64_t>(int32_value->value()));
}

inline AnfNodePtr ConvertFp32NodeToFp64(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  return ConvertScalarDtype(func_graph, node, kFloat32, kFloat64, ConvertFp32ValueToFp64);
}

inline AnfNodePtr ConvertInt32NodeToInt64(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  return ConvertScalarDtype(func_graph, node, kInt32, kInt64, ConvertInt32ValueToInt64);
}

AnfNodePtr ConvertTupleNodeDtype(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const TypePtr &from_type,
                                 const TypePtr &to_type,
                                 const std::function<ValuePtr(const ValuePtr &)> &convert_func) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  if (abs_tuple == nullptr) {
    MS_LOG(INFO) << "It should be an AbstractTuple, but is: " << abs->ToString();
    return nullptr;
  }
  if (abs_tuple->empty()) {
    return nullptr;
  }

  const TypePtrList &types = abs_tuple->ElementsType();
  static auto IsSourceType = [&from_type](const TypePtr &tp) { return tp == from_type; };
  if (!std::all_of(types.begin(), types.end(), IsSourceType)) {
    if (std::any_of(types.begin(), types.end(), IsSourceType)) {
      MS_LOG(INFO) << "The tuple contains elements of different dtypes! Can not handle it! " << abs->ToString();
    }
    return nullptr;
  }
  MS_LOG(DEBUG) << "Convert Tuple[" << from_type->ToString() << "] to Tuple[" << to_type->ToString()
                << "], node: " << node->DebugString() << ", abstract: " << abs->ToString();
  auto value = abs_tuple->GetValue();
  MS_EXCEPTION_IF_NULL(value);
  if (value->ContainsValueAny()) {
    ValueNodePtr dtype_node = NewDtypeValueNode(to_type->type_id());
    // Subsequent infer_mindir pass will generate Abstract for these two nodes.
    auto tuple_to_tensor_node = func_graph->NewCNode({NewValueNode(prim::kPrimTupleToTensor), node, dtype_node});
    auto tensor_to_tuple_node = func_graph->NewCNode({NewValueNode(prim::kPrimTensorToTuple), tuple_to_tensor_node});
    return tensor_to_tuple_node;
  } else {
    ValuePtr new_tuple_value = convert_func(value);
    auto new_tuple_node = NewValueNode(new_tuple_value);
    new_tuple_node->set_abstract(new_tuple_value->ToAbstract());
    return new_tuple_node;
  }
}

ValuePtr ConvertTupleValue(const ValuePtr &value, const std::function<ValuePtr(const ValuePtr &)> &convert_func) {
  auto tuple_value = value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_value);
  std::vector<ValuePtr> new_elements;
  (void)std::transform(tuple_value->value().begin(), tuple_value->value().end(), std::back_inserter(new_elements),
                       convert_func);
  return std::make_shared<ValueTuple>(new_elements);
}

inline ValuePtr ConvertTupleOfFp32ValueToFp64(const ValuePtr &value) {
  return ConvertTupleValue(value, ConvertFp32ValueToFp64);
}

inline ValuePtr ConvertTupleOfInt32ValueToInt64(const ValuePtr &value) {
  return ConvertTupleValue(value, ConvertInt32ValueToInt64);
}

inline AnfNodePtr ConvertFp32TupleNodeToFp64(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  return ConvertTupleNodeDtype(func_graph, node, kFloat32, kFloat64, ConvertTupleOfFp32ValueToFp64);
}

inline AnfNodePtr ConvertInt32TupleNodeToInt64(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  return ConvertTupleNodeDtype(func_graph, node, kInt32, kInt64, ConvertTupleOfInt32ValueToInt64);
}

bool ConvertPrimitiveArgumentDtype(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  ops::OpDefPtr op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def == nullptr) {
    // It is an old primitive (without xxx_op.yaml) that supports all scalar types, no need to convert, it's okay.
    return false;
  }
  if (cnode->size() <= 1) {
    return false;
  }

  std::map<size_t, AnfNodePtr> new_args;
  TraceGuard trace_guard(MakeTraceInfo<TraceOpt>(cnode->debug_info()));
  ScopeGuard scope_guard(cnode->scope());
  size_t sz = std::min(op_def->args_.size(), cnode->size() - 1);
  for (size_t i = 0; i < sz; ++i) {
    const ops::OpInputArg &arg = op_def->args_[i];
    auto arg_node = cnode->input(i + 1);
    MS_EXCEPTION_IF_NULL(arg_node);
    if (arg_node->abstract() == nullptr) {
      continue;
    }
    AnfNodePtr new_arg_node;
    if (arg.arg_dtype_ == ops::DT_FLOAT) {
      new_arg_node = ConvertFp32NodeToFp64(func_graph, arg_node);
    } else if (arg.arg_dtype_ == ops::DT_INT) {
      new_arg_node = ConvertInt32NodeToInt64(func_graph, arg_node);
    } else if (arg.arg_dtype_ == ops::DT_TUPLE_FLOAT) {
      new_arg_node = ConvertFp32TupleNodeToFp64(func_graph, arg_node);
    } else if (arg.arg_dtype_ == ops::DT_TUPLE_INT) {
      new_arg_node = ConvertInt32TupleNodeToInt64(func_graph, arg_node);
    }
    if (new_arg_node != nullptr) {
      (void)new_args.emplace(i, new_arg_node);
    }
  }
  if (new_args.empty()) {
    return false;
  }

  AnfNodePtrList new_inputs{cnode->inputs().begin(), cnode->inputs().end()};
  for (const auto &[arg_idx, new_arg_node] : new_args) {
    new_inputs[arg_idx + 1] = new_arg_node;
  }
  auto new_cnode = func_graph->NewCNode(new_inputs);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_attrs(cnode->attrs());
  MS_LOG(DEBUG) << "Argument dtype is converted, replace old cnode with new one. old: " << cnode->DebugString()
                << ", new: " << new_cnode->DebugString();
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(cnode, new_cnode);
  return true;
}

inline bool IsFp32Scalar(const ValuePtr &value) { return value != nullptr && value->isa<FP32Imm>(); }

inline bool IsInt32Scalar(const ValuePtr &value) { return value != nullptr && value->isa<Int32Imm>(); }

template <typename T, typename = std::enable_if_t<std::is_base_of_v<Scalar, T>>>
bool IsTupleOfScalar(const ValuePtr &value) {
  if (value == nullptr || !value->isa<ValueTuple>()) {
    return false;
  }
  auto tuple_value = value->cast_ptr<ValueTuple>();
  MS_EXCEPTION_IF_NULL(tuple_value);
  const ValuePtrList &elements = tuple_value->value();
  return !elements.empty() && std::all_of(elements.begin(), elements.end(),
                                          [](const ValuePtr &elem) { return elem != nullptr && elem->isa<T>(); });
}

inline bool NeedConvertDtype(const ValuePtr &value) {
  return IsFp32Scalar(value) || IsInt32Scalar(value) || IsTupleOfScalar<FP32Imm>(value) ||
         IsTupleOfScalar<Int32Imm>(value);
}

void ConvertPrimitiveAttributeDtype(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!std::any_of(prim->attrs().begin(), prim->attrs().end(),
                   [](const auto &kv) { return NeedConvertDtype(kv.second); })) {
    return;
  }
  std::vector<std::pair<std::string, ValuePtr>> new_attrs;
  for (const auto &[key, value] : prim->attrs()) {
    if (IsFp32Scalar(value)) {
      new_attrs.emplace_back(key, ConvertFp32ValueToFp64(value));
    } else if (IsInt32Scalar(value)) {
      new_attrs.emplace_back(key, ConvertInt32ValueToInt64(value));
    } else if (IsTupleOfScalar<FP32Imm>(value)) {
      new_attrs.emplace_back(key, ConvertTupleOfFp32ValueToFp64(value));
    } else if (IsTupleOfScalar<Int32Imm>(value)) {
      new_attrs.emplace_back(key, ConvertTupleOfInt32ValueToInt64(value));
    }
  }
  prim->SetAttrs(new_attrs);
  MS_LOG(DEBUG) << "Primitive '" << prim->ToString() << "' converts " << new_attrs.size() << " attributes' dtype";
}
}  // namespace

bool ConvertScalarDtypeForLegacyMindIR(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag("skip_converting_mindir_scalar_dtype")) {
    // The old version MindIR (mindspore version < 2.7.0) didn't have this flag.
    MS_LOG(DEBUG) << "No need to convert scalar dtype, skip this stage";
    return false;
  }
  MS_LOG(INFO) << "It's an old version MindIR and requires scalar dtype conversion";

  int cnt = 0;
  const AnfNodePtrList &all_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }

    ConvertPrimitiveAttributeDtype(prim);
    bool converted = ConvertPrimitiveArgumentDtype(func_graph, cnode, prim);
    cnt += (converted ? 1 : 0);
  }
  MS_LOG(INFO) << cnt << " nodes dtype have been converted.";
  return cnt > 0;
}

}  // namespace pipeline
}  // namespace mindspore
