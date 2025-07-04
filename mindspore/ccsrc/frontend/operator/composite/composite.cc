/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/composite.h"
#include <algorithm>
#include <string>
#include <tuple>
#include <regex>
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "utils/symbolic.h"
#include "include/common/fallback.h"
#include "include/common/pybind_api/api_register.h"
#include "ir/signature.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "utils/interpret_node_recorder.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
constexpr auto kStepDefault = 1;

using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractBasePtr;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractElementPair;
using mindspore::abstract::AbstractEllipsis;
using mindspore::abstract::AbstractEllipsisPtr;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractListPtr;
using mindspore::abstract::AbstractNone;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractSlice;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::abstract::AbstractUndetermined;
using mindspore::abstract::EnvSetSparseResultMgr;
using mindspore::abstract::FuncGraphAbstractClosure;
using mindspore::abstract::PartialAbstractClosure;

void HyperMap::Init() {
  if (fn_leaf_) {
    name_ = "hyper_map[" + fn_leaf_->name() + "]";
  }
  signatures_ =
    // def hypermap(func:read, *args:ref):
    std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                            {"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
}

HyperMap::HyperMap(bool reverse, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf)
    : MetaFuncGraph("hyper_map"),
      fn_leaf_(fn_leaf),
      reverse_(reverse),
      nonleaf_({kObjectTypeList, kObjectTypeTuple, kObjectTypeDictionary}) {
  Init();
}

HyperMap::HyperMap(const HyperMap &h)
    : MetaFuncGraph("hyper_map"), fn_leaf_(h.fn_leaf_), reverse_(h.reverse_), nonleaf_(h.nonleaf_) {
  Init();
}

void HyperMap::SetObjectForFnLeaf(const py::object &leaf_object) {
  if (fn_leaf_ != nullptr) {
    fn_leaf_->set_meta_obj(leaf_object);
  }
}

AnfNodePtr HyperMap::FullMake(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                              const ArgsPairList &arg_map) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> inputs;
  if (fn_arg != nullptr) {
    inputs.push_back(fn_arg);
  } else {
    inputs.push_back(NewValueNode(fn_leaf_));
  }

  (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs),
                       [](const std::pair<AnfNodePtr, Any> &item) { return item.first; });
  return func_graph->NewCNodeInOrder(inputs);
}

std::pair<std::string, std::string> HyperMap::GetHyperMapInputIndex(size_t num) const {
  std::string error_index;
  std::string next_index;
  const size_t first_index = 1;
  const size_t second_index = 2;
  if (num == first_index) {
    // The first element in HyperMap is func_graph
    error_index = "first";
    next_index = "second";
  } else if (num == second_index) {
    error_index = "second";
    next_index = "third";
  } else {
    error_index = std::to_string(num) + "th";
    next_index = std::to_string(num + 1) + "th";
  }
  return std::pair<std::string, std::string>(error_index, next_index);
}

template <typename T>
void HyperMap::CheckArgsInSequence(const ArgsPairList &arg_map, TypeId type_id, std::size_t size,
                                   bool *contains_dyn) const {
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_map) {
    num++;
    auto lhs = std::static_pointer_cast<T>(item.second);
    auto [error_index_res, next_index_res] = GetHyperMapInputIndex(num);
    if (lhs == nullptr) {
      std::string type_name = "List";
      if (type_id == kObjectTypeTuple) {
        type_name = "Tuple";
      }
      MS_LOG(EXCEPTION) << "The " << error_index_res << " element in HyperMap has wrong type, expected a " << type_name
                        << ", but got " << item.second->ToString() << ".";
    }
    if (lhs->dynamic_len()) {
      *contains_dyn = true;
      continue;
    }
    size_t ele_size = lhs->elements().size();
    if (ele_size != size) {
      oss << "\nThe length of the " << error_index_res << " element in HyperMap is " << size
          << ", but the length of the " << next_index_res << " element in HyperMap is " << ele_size << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    std::string types_name = "lists";
    if (type_id == kObjectTypeTuple) {
      types_name = "tuples";
    }
    MS_LOG(EXCEPTION) << "The " << types_name << " in HyperMap should have the same length. " << oss.str();
  }
}

AnfNodePtr HyperMap::HyperMapConverter(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                                       const ArgsPairList &arg_map, TypeId type_id, std::size_t size) const {
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  constexpr size_t kPrimHoldLen = 1;
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(size + kPrimHoldLen);
  if (type_id == kObjectTypeList) {
    inputs.push_back(NewValueNode(prim::kPrimMakeList));
  } else {
    inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  }
  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeList or FullMakeTuple for the " << i
                  << "th element of the target, reverse_: " << reverse_;
    std::vector<AnfNodePtr> inputs2;
    inputs2.push_back(fn_rec);
    if (fn_arg != nullptr) {
      inputs2.push_back(fn_arg);
    }
    size_t pos = (reverse_ ? (size - 1 - i) : i);
    (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(inputs2),
                         [&func_graph, &pos, &type_id](const std::pair<AnfNodePtr, Any> &item) {
                           if (type_id == kObjectTypeList) {
                             return func_graph->NewCNodeInOrder(
                               {NewValueNode(prim::kPrimListGetItem), item.first, NewValueNode(SizeToLong(pos))});
                           }
                           return func_graph->NewCNodeInOrder(
                             {NewValueNode(prim::kPrimTupleGetItem), item.first, NewValueNode(SizeToLong(pos))});
                         });

    auto call_node = func_graph->NewCNodeInOrder(inputs2);
    if (reverse_) {
      (void)inputs.insert(inputs.cbegin() + 1, call_node);
    } else {
      (void)inputs.emplace_back(call_node);
    }
  }
  if (inputs.size() > 1) {
    return func_graph->NewCNodeInOrder(inputs);
  }
  if (type_id == kObjectTypeList) {
    // Empty list.
    auto empty_list_value = std::make_shared<ValueList>(ValuePtrList());
    return NewValueNode(empty_list_value);
  }
  // Empty tuple.
  auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
  return NewValueNode(empty_tuple_value);
}

template <typename T>
AnfNodePtr HyperMap::HyperMapDynamicConverter(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                                              const ArgsPairList &arg_map, const TypePtr &element_type) const {
  TypeId type_id = std::make_shared<T>()->generic_type_id();
  MS_EXCEPTION_IF_NULL(element_type);
  if (element_type->isa<Tuple>() || element_type->isa<List>() || element_type->isa<Dictionary>()) {
    MS_EXCEPTION(TypeError) << "The HyperMap does not support scenarios involving nested dynamic " << type_id
                            << ", where the internal elements are " << element_type;
  }
  auto inner_fg = std::make_shared<FuncGraph>();
  auto func_input = inner_fg->add_parameter();
  const std::string module = "mindspore._extends.parse.standard_method";
  std::string func_name;
  std::vector<AnfNodePtr> ret_inputs;
  if (type_id == kObjectTypeList) {
    func_name = "hypermap_dynamic_list";
  } else {
    func_name = "hypermap_dynamic_tuple";
  }
  py::function fn = python_adapter::GetPyFn(module, func_name);
  auto prim_func = parse::ParsePythonCode(fn);
  (void)ret_inputs.insert(ret_inputs.end(), {NewValueNode(prim::kPrimMakeTuple), func_input});
  for (auto e : arg_map) {
    (void)ret_inputs.emplace_back(inner_fg->add_parameter());
  }
  auto ret_node = inner_fg->NewCNodeInOrder(ret_inputs);
  std::vector<AnfNodePtr> inner_ret_inputs = {NewValueNode(prim::kPrimDoUnpackCall), NewValueNode(prim_func), ret_node};
  auto inner_ret = inner_fg->NewCNodeInOrder(inner_ret_inputs);
  inner_fg->set_output(inner_ret);
  std::vector<AnfNodePtr> final_node_input = {NewValueNode(inner_fg)};
  if (fn_leaf_ == nullptr) {
    final_node_input.push_back(fn_arg);
  } else {
    final_node_input.push_back(NewValueNode(fn_leaf_));
  }
  (void)std::transform(arg_map.begin(), arg_map.end(), std::back_inserter(final_node_input),
                       [](const std::pair<AnfNodePtr, TypePtr> &item) { return item.first; });
  return func_graph->NewCNodeInOrder(final_node_input);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);
  if (type->dynamic_len()) {
    return HyperMapDynamicConverter<List>(func_graph, fn_arg, arg_map, type->dynamic_element_type());
  }
  size_t size = type->elements().size();
  bool contains_dynamic = false;
  CheckArgsInSequence<List>(arg_map, kObjectTypeList, size, &contains_dynamic);
  if (contains_dynamic) {
    return HyperMapDynamicConverter<List>(func_graph, fn_arg, arg_map, type->elements()[0]);
  }
  // Cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  return HyperMapConverter(func_graph, fn_arg, arg_map, kObjectTypeList, size);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);
  if (type->dynamic_len()) {
    return HyperMapDynamicConverter<Tuple>(func_graph, fn_arg, arg_map, type->dynamic_element_type());
  }
  size_t size = type->elements().size();
  bool contains_dynamic = false;
  CheckArgsInSequence<Tuple>(arg_map, kObjectTypeTuple, size, &contains_dynamic);
  if (contains_dynamic) {
    return HyperMapDynamicConverter<Tuple>(func_graph, fn_arg, arg_map, type->elements()[0]);
  }
  // Cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  return HyperMapConverter(func_graph, fn_arg, arg_map, kObjectTypeTuple, size);
}

AnfNodePtr HyperMap::FullMake(const std::shared_ptr<Dictionary> &type, const FuncGraphPtr &func_graph,
                              const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(type);

  size_t size = type->key_values().size();
  size_t num = 0;
  std::ostringstream oss;
  bool is_not_same = false;
  for (auto &item : arg_map) {
    num++;
    auto lhs = std::static_pointer_cast<Dictionary>(item.second);
    auto [dict_error_index, dict_next_index] = GetHyperMapInputIndex(num);
    if (lhs == nullptr) {
      MS_LOG(EXCEPTION) << "The " << dict_error_index
                        << " element in HyperMap has wrong type, expected a Dictionary, but got "
                        << item.second->ToString() << ".";
    }
    if (lhs->key_values().size() != size) {
      oss << "\nThe length of the " << dict_error_index << " element in HyperMap is " << size
          << ", but the length of the " << dict_next_index << " element in HyperMap is " << lhs->key_values().size()
          << ".\n";
      is_not_same = true;
      break;
    }
  }
  if (is_not_same) {
    MS_LOG(EXCEPTION) << "The length of dict in HyperMap must be the same. " << oss.str();
  }

  // cannot use shared_from_base() also known as this, as it will make a reference cycle on
  // hypermap and graph generated, it will cause memory leak.
  auto fn_rec = NewValueNode(std::make_shared<HyperMap>(*this));
  std::vector<AnfNodePtr> key_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> value_inputs{NewValueNode(prim::kPrimMakeTuple)};

  for (size_t i = 0; i < size; i++) {
    MS_LOG(DEBUG) << "FullMakeDict for the " << i << "th element of the target.";
    auto key = type->key_values()[i].first;
    (void)key_inputs.emplace_back(NewValueNode(key));
    std::vector<AnfNodePtr> inputs;
    (void)inputs.emplace_back(fn_rec);
    if (fn_arg != nullptr) {
      (void)inputs.emplace_back(fn_arg);
    }
    (void)std::transform(
      arg_map.begin(), arg_map.end(), std::back_inserter(inputs),
      [&func_graph, &key](const std::pair<AnfNodePtr, TypePtr> &item) {
        return func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimDictGetItem), item.first, NewValueNode(key)});
      });
    auto call_node = func_graph->NewCNodeInOrder(inputs);
    (void)value_inputs.emplace_back(call_node);
  }
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeDict), func_graph->NewCNodeInOrder(key_inputs),
                                 func_graph->NewCNodeInOrder(value_inputs)};
  return func_graph->NewCNodeInOrder(inputs);
}

AnfNodePtr HyperMap::Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const {
  bool is_leaf = false;
  TypeId id = kObjectTypeEnd;
  std::pair<AnfNodePtr, TypePtr> pair;
  for (auto &item : arg_map) {
    pair = item;
    id = item.second->type_id();
    // The graph building reaches the leaf situation when there exists type that can not be divided any more.
    if (nonleaf_.count(id) == 0) {
      is_leaf = true;
      break;
    }
  }

  if (!is_leaf) {
    // In a nonleaf situation, all arguments must have the same generic.
    bool is_not_same = std::any_of(arg_map.begin(), arg_map.end(), [pair](const std::pair<AnfNodePtr, TypePtr> &item) {
      if (item.first != pair.first) {
        return item.second->type_id() != pair.second->type_id();
      }
      return false;
    });
    if (is_not_same) {
      std::ostringstream oss;
      oss << "There are " << arg_map.size() << " inputs of `" << name_ << "`, corresponding type info:\n"
          << trace::GetDebugInfoStr(func_graph->debug_info()) << "\n";
      int64_t idx = 0;
      std::string str_index = "first";
      const int64_t diff_index = 2;
      for (auto &item : arg_map) {
        // The first element in HyperMap is func_graph
        if (idx == 0) {
          str_index = "second";
        } else if (idx == 1) {
          str_index = "third";
        } else {
          str_index = std::to_string(idx + diff_index) + "th";
        }
        ++idx;
        oss << "The type of the " << str_index << " argument in HyperMap is " << item.second->ToString() << ".\n";
      }
      MS_LOG(EXCEPTION) << "In a nonleaf situation, the types of arguments in HyperMap must be consistent, "
                        << "but the types of arguments are inconsistent.\n"
                        << oss.str();
    }
  }

  switch (id) {
    case kObjectTypeList: {
      auto type = std::static_pointer_cast<List>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    case kObjectTypeTuple: {
      auto type = std::static_pointer_cast<Tuple>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    case kObjectTypeDictionary: {
      auto type = std::static_pointer_cast<Dictionary>(pair.second);
      return FullMake(type, func_graph, fn_arg, arg_map);
    }
    default:
      return FullMake(func_graph, fn_arg, arg_map);
  }
}

FuncGraphPtr HyperMap::GenerateFromTypes(const TypePtrList &args_abs_list) {
  FuncGraphPtr res_fg = std::make_shared<FuncGraph>();
  res_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_fg->set_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER, true);
  if (res_fg->debug_info() != nullptr) {
    res_fg->debug_info()->set_name("hyper_map");
  }

  AnfNodePtr fn_param = nullptr;
  std::size_t i = 0;
  ArgsPairList argmap;
  if (fn_leaf_ == nullptr) {
    fn_param = res_fg->add_parameter();
    i = 1;
  }

  std::size_t size = args_abs_list.size();
  for (; i < size; ++i) {
    argmap.push_back(std::make_pair(res_fg->add_parameter(), args_abs_list[i]));
  }

  res_fg->set_output(Make(res_fg, fn_param, argmap));
  return res_fg;
}

abstract::AbstractBasePtrList HyperMap::NormalizeArgs(const AbstractBasePtrList &args_abs_list) const {
  if (fn_leaf_ == nullptr) {
    if (args_abs_list.empty()) {
      MS_LOG(EXCEPTION) << "The size of arguments in list should not be empty. But the size of arguments is 0.";
    }
    MS_EXCEPTION_IF_NULL(args_abs_list[0]);
    // Assert that hypermap's function param does not contain free variables
    if (args_abs_list[0]->isa<FuncGraphAbstractClosure>()) {
      auto graph_func = dyn_cast<FuncGraphAbstractClosure>(args_abs_list[0]);
      auto func_graph = graph_func->func_graph();
      if (func_graph->parent() != nullptr) {
        MS_LOG(EXCEPTION) << "HyperMap don't support Closure with free variable yet.";
      }
    }
  }

  AbstractBasePtrList broadened;
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(broadened),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->Broaden();
                       });
  return broadened;
}

FuncGraphPtr MakeTupleGradient::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  int64_t tuple_size = SizeToLong(args_abs_list.size());

  std::ostringstream ss;
  // ▶make_tuple_
  ss << "\u25B8make_tuple_" << tuple_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name(ss.str());
  }

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (int64_t i = 0; i < tuple_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // Make fprop first result, make_tuple's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // Make fprop second result, make_tuple's backward function.
  FuncGraphPtr bprop = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  // ◀make_tuple_
  ss << "\u25C2make_tuple_" << tuple_size;
  if (bprop->debug_info() != nullptr) {
    bprop->debug_info()->set_name(ss.str());
  }
  AnfNodePtr dout = bprop->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewEnviron(bprop));
  for (int64_t i = 0; i < tuple_size; ++i) {
    grads.push_back(bprop->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(i)}));
  }

  bprop->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop->set_output(bprop->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(bprop)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeTuple));
  return fg;
}

FuncGraphPtr MakeListGradient::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  int64_t list_size = SizeToLong(args_abs_list.size());

  std::ostringstream ss;
  // ▶make_list_
  ss << "\u25B8make_list_" << list_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name(ss.str());
  }

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMakeList));
  for (int64_t i = 0; i < list_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // Make fprop first result, make_list's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // Make fprop second result, make_list's backward function.
  FuncGraphPtr bprop = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  // ◀make_list_
  ss << "\u25C2make_list_" << list_size;
  if (bprop->debug_info() != nullptr) {
    bprop->debug_info()->set_name(ss.str());
  }
  AnfNodePtr dout = bprop->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewEnviron(bprop));
  for (int64_t i = 0; i < list_size; ++i) {
    grads.push_back(bprop->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), dout, NewValueNode(i)}));
  }

  bprop->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop->set_output(bprop->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(bprop)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeList));
  return fg;
}

FuncGraphPtr MakeDictGradient::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 2;
  CheckArgsSize("MakeDict", args_abs_list, input_size);
  std::ostringstream ss;
  // ▶make_dict_
  ss << "\u25B8make_dict_" << input_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name(ss.str());
  }

  std::vector<AnfNodePtr> params{NewValueNode(prim::kPrimMakeDict)};
  for (size_t i = 0; i < input_size; ++i) {
    (void)params.emplace_back(fg->add_parameter());
  }

  // Make fprop first result, make_dict's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // Make fprop second result, make_dict's backward function.
  FuncGraphPtr bprop = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  // ◀make_dict_
  ss << "\u25C2make_dict_" << input_size;
  if (bprop->debug_info() != nullptr) {
    bprop->debug_info()->set_name(ss.str());
  }
  AnfNodePtr dout = bprop->add_parameter();

  std::vector<AnfNodePtr> grads{NewValueNode(prim::kPrimMakeTuple)};
  (void)grads.emplace_back(NewEnviron(bprop));

  auto abs0_tuple = dyn_cast_ptr<AbstractTuple>(args_abs_list[0]);
  if (abs0_tuple == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The first input of make_dict should be a tuple, but got abstract: "
                               << args_abs_list[0]->ToString();
  }
  // Add gradients of keys tuple and values tuple.
  std::vector<AnfNodePtr> keys_grads_inputs{NewValueNode(kPrimMakeTuple)};
  std::vector<AnfNodePtr> values_grads_inputs{NewValueNode(kPrimMakeTuple)};
  for (size_t i = 0; i < abs0_tuple->size(); ++i) {
    auto key_item =
      bprop->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), params[1], NewValueNode(SizeToLong(i))});
    (void)keys_grads_inputs.emplace_back(key_item);
    (void)values_grads_inputs.emplace_back(
      bprop->NewCNodeInOrder({NewValueNode(prim::kPrimDictGetItem), dout, key_item}));
  }
  (void)grads.emplace_back(bprop->NewCNodeInOrder(keys_grads_inputs));
  (void)grads.emplace_back(bprop->NewCNodeInOrder(values_grads_inputs));

  bprop->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop->set_output(bprop->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(bprop)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMakeDict));
  return fg;
}

FuncGraphPtr PyExecuteGradient::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  int64_t args_size = SizeToLong(args_abs_list.size());
  constexpr auto py_execute_grad_input_count = 3;
  if (args_size < py_execute_grad_input_count) {
    MS_LOG(INTERNAL_EXCEPTION) << "The inputs size of PyExecuteGradient should not less than "
                               << py_execute_grad_input_count;
  }

  std::ostringstream ss;
  // ▶PyExecute
  ss << "\u25B8PyExecute_" << args_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name(ss.str());
  }

  std::vector<AnfNodePtr> params;
  (void)params.emplace_back(NewValueNode(prim::kPrimPyExecute));
  for (int64_t i = 0; i < args_size; ++i) {
    (void)params.emplace_back(fg->add_parameter());
  }

  // Make fprop first result, PyExecute's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);
  InterpretNodeRecorder::GetInstance().PushPyExecuteNode(out);

  // Make fprop second result, PyExecute's backward function.
  FuncGraphPtr bprop = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  // ◀PyExecute
  ss << "\u25C2PyExecute_" << args_size;
  if (bprop->debug_info() != nullptr) {
    bprop->debug_info()->set_name(ss.str());
  }
  (void)bprop->add_parameter();

  std::vector<AnfNodePtr> grads;
  (void)grads.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  (void)grads.emplace_back(NewEnviron(bprop));
  // Propagate for script string.
  (void)grads.emplace_back(params[1]);
  // Propagate for local dict keys.
  const auto &local_key_args = dyn_cast<abstract::AbstractTuple>(args_abs_list[1]);
  MS_EXCEPTION_IF_NULL(local_key_args);
  std::vector<AnfNodePtr> keys;
  (void)keys.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t i = 0; i < local_key_args->size(); ++i) {
    constexpr auto keys_num = 2;
    const auto &key_item =
      bprop->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), params[keys_num], NewValueNode(SizeToLong(i))});
    const auto &element = local_key_args->elements()[i];
    const auto &str_element = dyn_cast<abstract::AbstractScalar>(element);
    if (str_element != nullptr && str_element->BuildType()->isa<String>()) {
      (void)keys.emplace_back(key_item);
    } else {
      (void)keys.emplace_back(bprop->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("zeros_like")), key_item}));
    }
  }
  (void)grads.emplace_back(bprop->NewCNodeInOrder(keys));
  // Propagate for local dict values.
  constexpr auto values_arg_num = 2;
  const auto &local_value_args = dyn_cast<abstract::AbstractTuple>(args_abs_list[values_arg_num]);
  MS_EXCEPTION_IF_NULL(local_value_args);
  std::vector<AnfNodePtr> values;
  (void)values.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t i = 0; i < local_value_args->size(); ++i) {
    constexpr auto values_num = 3;
    const auto &value_item =
      bprop->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), params[values_num], NewValueNode(SizeToLong(i))});
    const auto &element = local_value_args->elements()[i];
    const auto &str_element = dyn_cast<abstract::AbstractScalar>(element);
    if (str_element != nullptr && str_element->BuildType()->isa<String>()) {
      (void)values.emplace_back(value_item);
    } else {
      (void)values.emplace_back(bprop->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("zeros_like")), value_item}));
    }
  }
  (void)grads.emplace_back(bprop->NewCNodeInOrder(values));

  // Add gradients for extra monad.
  for (size_t i = py_execute_grad_input_count; i < args_abs_list.size(); ++i) {
    if (args_abs_list[i]->isa<abstract::AbstractUMonad>()) {
      (void)grads.emplace_back(NewValueNode(kUMonad));
    } else if (args_abs_list[i]->isa<abstract::AbstractIOMonad>()) {
      (void)grads.emplace_back(NewValueNode(kIOMonad));
    } else {
      (void)grads.emplace_back(NewValueNode(kValueAny));
    }
  }

  bprop->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop->set_output(bprop->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(bprop)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimPyExecute));
  return fg;
}

FuncGraphPtr MutableGradient::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t min_input_size = 1;
  constexpr size_t max_input_size = 2;
  auto input_size = args_abs_list.size();
  if (input_size != min_input_size && input_size != max_input_size) {
    MS_LOG(EXCEPTION) << "The number of input to mutable must be " << min_input_size << " or " << max_input_size
                      << ", but got: " << input_size;
  }
  std::ostringstream ss;
  // ▶mutable_
  ss << "\u25B8mutable_" << input_size;
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name(ss.str());
  }

  std::vector<AnfNodePtr> params;
  params.push_back(NewValueNode(prim::kPrimMutable));
  for (size_t i = 0; i < input_size; ++i) {
    params.push_back(fg->add_parameter());
  }

  // Make fprop first result, mutable's forward result.
  AnfNodePtr out = fg->NewCNodeInOrder(params);

  // Make fprop second result, mutable's backward function.
  FuncGraphPtr bprop = std::make_shared<FuncGraph>();

  ss.str(std::string());
  ss.clear();
  // ◀mutable_
  ss << "\u25C2mutable_" << input_size;
  if (bprop->debug_info() != nullptr) {
    bprop->debug_info()->set_name(ss.str());
  }
  AnfNodePtr dout = bprop->add_parameter();

  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(prim::kPrimMakeTuple));
  grads.push_back(NewEnviron(bprop));
  grads.push_back(dout);
  if (input_size == max_input_size) {
    grads.push_back(bprop->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("zeros_like")), params[2]}));
  }

  bprop->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop->set_output(bprop->NewCNodeInOrder(grads));

  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->set_output(fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), out, NewValueNode(bprop)}));
  (void)fg->transforms().emplace("primal", FuncGraphTransform(prim::kPrimMutable));
  return fg;
}

namespace {
bool IsTupleAllTensor(const AbstractTuplePtr &tuple_arg) {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  for (size_t i = 0; i < tuple_arg->size(); ++i) {
    if (!(*tuple_arg)[i]->isa<AbstractUndetermined>() &&
        !((*tuple_arg)[i]->isa<AbstractTuple>() && IsTupleAllTensor((*tuple_arg)[i]->cast<AbstractTuplePtr>()))) {
      return false;
    }
  }
  return true;
}

bool EnableGradFirstForTuple(const AbstractTuplePtr &tuple_arg, bool enable_tuple_grad) {
  return tuple_arg->size() > 1 && (*tuple_arg)[1]->isa<AbstractTuple>() && enable_tuple_grad &&
         IsTupleAllTensor((*tuple_arg)[1]->cast<AbstractTuplePtr>());
}

bool EnableGradForScalar(const AbstractBasePtr &abs) {
  return (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1") &&
         abs->BuildType() != nullptr && abs->BuildType()->isa<Number>();
}

bool CanGradArgument(const AbstractTuplePtr &tuple_arg, size_t pos) {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  return tuple_arg->size() > pos && (*tuple_arg)[pos] != nullptr &&
         (((*tuple_arg)[pos]->BuildValue() != nullptr && (*tuple_arg)[pos]->BuildValue()->ContainsValueAny()) ||
          EnableGradForScalar((*tuple_arg)[pos]));
}

void GenerateFuncGraphByPosition(const FuncGraphPtr &fg, const AbstractTuplePtr &tuple_arg, const AbstractTuplePtr &pos,
                                 bool return_ids = false) {
  if (pos == nullptr) {
    MS_LOG(EXCEPTION) << "Return grad by position, but the grad_position is empty!";
  }
  if (pos->empty()) {
    MS_LOG(EXCEPTION) << "grad_position should not be empty when grad by position.";
  }
  AnfNodePtr tuple_parameter = fg->add_parameter();
  (void)fg->add_parameter();  // The 'grad_position' parameter.
  // Collect all parameters by 'grad_position'.
  std::vector<AnfNodePtr> pos_elements = {NewValueNode(prim::kPrimMakeTuple)};
  CNodePtr current_element = nullptr;
  for (size_t i = 0; i < pos->size(); ++i) {
    auto val = pos->elements()[i]->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    auto int_val = LongToSize(dyn_cast<Int64Imm>(val)->value());
    ++int_val;  // Ignore the env position.
    if (int_val >= tuple_arg->size()) {
      MS_EXCEPTION(IndexError) << "Position index " << (int_val - 1) << " is exceed input size.";
    }
    if (!CanGradArgument(tuple_arg, int_val)) {
      continue;
    }
    current_element =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(int_val))});
    if (return_ids) {
      current_element =
        fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(SizeToLong(int_val) - 1), current_element});
    }
    pos_elements.push_back(current_element);
  }

  // The returned result may vary for grad result element number.
  // A single value if only one result, a tuple for multiple results, or a empty tuple for no result.
  //
  // Notice that even if the user set 'grad_position' as multiple choices,
  // the 'CanGradArgument' may change it to only one choice or none choice.
  constexpr size_t args_least_size = 2;
  if (pos_elements.size() == args_least_size) {
    fg->set_output(current_element);
  } else if (pos_elements.size() > args_least_size) {
    fg->set_output(fg->NewCNodeInOrder(pos_elements));
  } else {  // The 'pos' is empty AbstractTuple.
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
  }
}
}  // namespace

FuncGraphPtr Tail::GenerateTailFuncGraph(const AbstractSequencePtr &sequence_arg) const {
  MS_EXCEPTION_IF_NULL(sequence_arg);
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name("tail");
  }

  AnfNodePtr tuple_parameter = fg->add_parameter();
  std::vector<AnfNodePtr> elements;
  PrimitivePtr op = nullptr;
  if (sequence_arg->isa<AbstractTuple>()) {
    (void)elements.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    op = prim::kPrimTupleGetItem;
  } else {
    (void)elements.emplace_back(NewValueNode(prim::kPrimMakeList));
    op = prim::kPrimListGetItem;
  }

  // Remove the first element to make a new sequence.
  for (size_t i = 1; i < sequence_arg->size(); ++i) {
    elements.push_back(fg->NewCNodeInOrder({NewValueNode(op), tuple_parameter, NewValueNode(SizeToLong(i))}));
  }
  if (elements.size() > 1) {
    fg->set_output(fg->NewCNodeInOrder(elements));
    return fg;
  }

  // No element left, return empty tuple.
  if (sequence_arg->isa<AbstractTuple>()) {
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
  }
  // No element left, return empty list.
  auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
  auto empty_tuple = NewValueNode(empty_tuple_value);
  fg->set_output(empty_tuple);
  return fg;
}

FuncGraphPtr Tail::GenerateGradFuncGraph(const AbstractTuplePtr &tuple_arg, const AbstractTuplePtr &position) const {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name("grad_tail");
  }

  if (tail_type_ == kGradFirst) {
    AnfNodePtr tuple_parameter = fg->add_parameter();
    if (CanGradArgument(tuple_arg, 1) || EnableGradFirstForTuple(tuple_arg, enable_tuple_grad_first_)) {
      fg->set_output(
        fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(1))}));
    } else {
      fg->set_output(NewValueNode(std::make_shared<ValueTuple>(ValuePtrList())));
    }
    return fg;
  }

  if (tail_type_ == kGradByPosition) {
    GenerateFuncGraphByPosition(fg, tuple_arg, position, return_ids_);
    return fg;
  }

  if (tail_type_ == kGradAll) {
    AnfNodePtr tuple_parameter = fg->add_parameter();
    std::vector<AnfNodePtr> elements = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 1; i < tuple_arg->size(); ++i) {
      MS_EXCEPTION_IF_NULL((*tuple_arg)[i]);
      if (CanGradArgument(tuple_arg, i)) {
        elements.push_back(
          fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(i))}));
      }
    }

    // We should deal with 'get_all=True' as other options later:
    // "The returned result may vary for grad result element number.
    // A single value if only one result, a tuple for multiple results, or a empty tuple for no result.
    //
    // Notice that even if the user set 'get_all=True' and pass multiple inputs,
    // the 'CanGradArgument' may change it to only one gradient output or no gradient."
    constexpr size_t args_least_size = 2;
    if (elements.size() >= args_least_size) {
      fg->set_output(fg->NewCNodeInOrder(elements));
      return fg;
    }
    // Empty tuple.
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
    return fg;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "'tail_type_' is not for GradOperation, but " << tail_type_;
}

FuncGraphPtr Tail::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // To handle normal tail.
  if (args_abs_list.size() < 1) {
    MS_LOG(EXCEPTION) << "'Tail' requires at least 1 argument, but got " << args_abs_list.size();
  }
  if (tail_type_ >= kNotGrad) {
    AbstractSequencePtr sequence_arg = dyn_cast<AbstractSequence>(args_abs_list[0]);
    if (sequence_arg == nullptr) {
      MS_LOG(EXCEPTION) << "'Tail' arg0 must be tuple or list, but got " << args_abs_list[0]->ToString();
    }
    return GenerateTailFuncGraph(sequence_arg);
  }

  // To handle for GradOperation tail.
  constexpr size_t args_max_size = 2;
  if (args_abs_list.size() > args_max_size) {
    MS_LOG(EXCEPTION) << "'Tail' requires at most 2 arguments for GradOperation, but got " << args_abs_list.size();
  }
  AbstractTuplePtr tuple_arg = dyn_cast<AbstractTuple>(args_abs_list[0]);
  if (tuple_arg == nullptr) {
    MS_LOG(EXCEPTION) << "'Tail' arg0 must be tuple, but got " << args_abs_list[0]->ToString();
  }
  if (args_abs_list.size() == args_max_size) {
    AbstractTuplePtr pos = dyn_cast<AbstractTuple>(args_abs_list[1]);
    if (pos == nullptr) {
      MS_LOG(EXCEPTION) << "'Tail' arg1 'position' must be tuple, but got " << args_abs_list[1]->ToString();
    }
    return GenerateGradFuncGraph(tuple_arg, pos);
  }
  return GenerateGradFuncGraph(tuple_arg);
}
namespace {
AnfNodePtr CreateGradOutputs(const FuncGraphPtr &k_child, const AnfNodePtr &gradient, const AnfNodePtr &f_app,
                             bool has_aux, bool get_value) {
  if (get_value) {
    return k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), f_app, gradient});
  }
  if (!has_aux) {
    return gradient;
  }
  PrimitivePtr get_tuple_item_op = prim::kPrimTupleGetItem;
  PrimitivePtr make_tuple_op = prim::kPrimMakeTuple;
  std::vector<AnfNodePtr> elements = {NewValueNode(make_tuple_op)};
  (void)elements.emplace_back(
    k_child->NewCNodeInOrder({NewValueNode(get_tuple_item_op), f_app, NewValueNode(static_cast<int64_t>(1))}));
  auto aux_output = k_child->NewCNodeInOrder(elements);
  auto unpack_node =
    k_child->NewCNodeInOrder({NewValueNode(get_tuple_item_op), aux_output, NewValueNode(static_cast<int64_t>(0))});
  return k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), gradient, unpack_node});
}
}  // namespace

// When set aux True, for out1, out2, out3 = fn(inputs), only first out1 contributes to differentiation of fn.
FuncGraphPtr GradAux::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  AbstractTuplePtr tuple_arg = dyn_cast<AbstractTuple>(args_abs_list[0]);
  if (tuple_arg == nullptr) {
    MS_LOG(EXCEPTION) << "When has_aux is True, origin fn requires more than one outputs.\n"
                      << "'GradAux' arg0 must be tuple, but got " << args_abs_list[0]->ToString();
  }
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr tuple_parameter = fg->add_parameter();
  // get_value flag
  (void)fg->add_parameter();

  AbstractScalarPtr get_value_ptr = dyn_cast<AbstractScalar>(args_abs_list[1]);
  MS_EXCEPTION_IF_NULL(get_value_ptr);
  bool get_value_flag = GetValue<bool>(get_value_ptr->BuildValue());
  std::vector<AnfNodePtr> elements = {NewValueNode(prim::kPrimMakeTuple)};
  elements.push_back(
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(0))}));
  if (get_value_flag) {
    for (size_t i = 1; i < tuple_arg->size(); i++) {
      auto aux_node =
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(i))});
      auto stop_gradient_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimStopGradient), aux_node});
      elements.push_back(stop_gradient_node);
    }
  } else {
    std::vector<AnfNodePtr> aux_elements = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 1; i < tuple_arg->size(); i++) {
      auto aux_node =
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(i))});
      auto stop_gradient_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimStopGradient), aux_node});
      aux_elements.push_back(stop_gradient_node);
    }
    elements.push_back(fg->NewCNodeInOrder(aux_elements));
  }

  constexpr size_t args_least_size = 2;
  if (elements.size() < args_least_size) {
    MS_LOG(EXCEPTION) << "When has_aux is True, origin fn requires more than one outputs, but got " << elements.size()
                      << " outputs.\n"
                      << trace::GetDebugInfoStr(fg->debug_info());
  }
  fg->set_output(fg->NewCNodeInOrder(elements));
  return fg;
}

GradOperation::GradOperation(const std::string &name, bool get_all, bool get_by_list, bool sens_param,
                             bool get_by_position, bool has_aux, bool get_value, bool return_ids, bool merge_forward)
    : MetaFuncGraph(name),
      get_all_(get_all),
      get_by_list_(get_by_list),
      sens_param_(sens_param),
      get_by_position_(get_by_position),
      has_aux_(has_aux),
      get_value_(get_value),
      return_ids_(return_ids),
      merge_forward_(merge_forward) {
  if (get_by_position) {
    signatures_ =
      // def grad(func:read, weight_list:ref, position_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault},
                              {"position_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  } else if (get_by_list) {
    signatures_ =
      // def grad(func:read, weight_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  }
}

FuncGraphPtr GradOperation::GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                                    const FuncGraphPtr &forward_graph, bool is_weights_none) const {
  FuncGraphPtr k_child = std::make_shared<FuncGraph>();
  k_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  k_child->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, true);

  AnfNodePtr position_node = nullptr;
  if (position != nullptr) {
    position_node = position;
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(j);
  for (size_t i = 0; i < forward_graph->parameters().size(); ++i) {
    inputs.push_back(k_child->add_parameter());
  }
  auto k_app = k_child->NewCNodeInOrder(inputs);

  auto tuple_get_item = NewValueNode(prim::kPrimTupleGetItem);
  auto f_app = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(0))});
  auto bprop = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(1))});

  GradByParameter(k_child, f_app, bprop, weights, position_node, forward_graph, is_weights_none);
  return k_child;
}

CNodePtr GradOperation::SetNodeByParameter(const CNodePtr &grad, const FuncGraphPtr &fg) const {
  CNodePtr fv_bprop;
  if (!weight_value_->isa<AbstractTuple>()) {
    auto weight_ref = dyn_cast<abstract::AbstractRefTensor>(weight_value_);
    if (weight_ref != nullptr) {
      auto weight_key = weight_ref->ref_key_value()->cast<RefKeyPtr>();
      auto param_name = weight_key->value();
      fv_bprop = fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(param_name), grad});
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Abstract of parameter should be AbstractRefTensor, but got "
                                 << weight_value_->ToString();
    }
  } else {
    std::vector<AnfNodePtr> params;
    AbstractTuplePtr weight_tuple = weight_value_->cast<AbstractTuplePtr>();
    const AbstractBasePtrList &elements = weight_tuple->elements();
    params.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < weight_tuple->size(); i++) {
      auto weight_ref = dyn_cast<abstract::AbstractRefTensor>(elements[i]);
      if (weight_ref != nullptr) {
        auto weight_key = weight_ref->ref_key_value()->cast<RefKeyPtr>();
        MS_EXCEPTION_IF_NULL(weight_key);
        auto param_name = weight_key->value();
        auto grad_value =
          fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), grad, NewValueNode(static_cast<int64_t>(i))});
        fv_bprop = fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(param_name), grad_value});
        params.push_back(fv_bprop);
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Abstract of parameter should be AbstractRefTensor, but got "
                                   << weight_value_->ToString();
      }
    }
    fv_bprop = fg->NewCNodeInOrder(params);
  }
  return fv_bprop;
}

// Do grad by the parameter of GradOperation.
void GradOperation::GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                                    const AnfNodePtr &weights, const AnfNodePtr &position,
                                    const FuncGraphPtr &forward_graph, bool is_weights_none) const {
  MS_EXCEPTION_IF_NULL(k_child);

  AnfNodePtr bprop_arg = nullptr;
  if (sens_param_) {
    bprop_arg = k_child->add_parameter();
  } else {
    auto ones_like = prim::GetPythonOps("_ones_like_for_grad");
    bprop_arg = k_child->NewCNodeInOrder({NewValueNode(ones_like), f_app});
  }
  AnfNodePtr b_app = k_child->NewCNodeInOrder({bprop, bprop_arg});
  // Add sense parameter flag for bound_node_.
  if (b_app->isa<CNode>() && sens_param_) {
    b_app->cast<CNodePtr>()->AddAttr("sens_param_", MakeValue(true));
  }

  CNodePtr fv_bprop = nullptr;
  if (get_by_list_) {
    if (is_weights_none) {
      fv_bprop = k_child->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple)});
    } else {
      // Python code: grads = hyper_map(F.partial(env_get, env), weights)
      AnfNodePtr env =
        k_child->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), b_app, NewValueNode(static_cast<int64_t>(0))});
      AnfNodePtr partial_env_get =
        k_child->NewCNodeInOrder({NewValueNode(prim::kPrimPartial), NewValueNode(prim::GetPythonOps("env_get")), env});
      MetaFuncGraphPtr hyper_map = std::make_shared<HyperMap>();
      fv_bprop = k_child->NewCNodeInOrder({NewValueNode(hyper_map), partial_env_get, weights});
      if (return_ids_) {
        fv_bprop = SetNodeByParameter(fv_bprop, k_child);
      }
    }
  }

  CNodePtr inputs_bprop = nullptr;
  if (get_by_position_) {
    TailPtr tail_grad_by_position = std::make_shared<Tail>("tail_grad_by_position", kGradByPosition, return_ids_);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_by_position), b_app, position});
  } else if (get_all_) {
    TailPtr tail_grad_all = std::make_shared<Tail>("tail_grad_all", kGradAll);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_all), b_app});
  }

  // Gradients wrt inputs and parameters
  if (fv_bprop != nullptr && inputs_bprop != nullptr) {
    auto make_tuple = k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), inputs_bprop, fv_bprop});
    k_child->set_output(CreateGradOutputs(k_child, make_tuple, f_app, has_aux_, get_value_));
    return;
  }

  // Gradients wrt parameters
  if (fv_bprop != nullptr) {
    k_child->set_output(CreateGradOutputs(k_child, fv_bprop, f_app, has_aux_, get_value_));
    return;
  }

  // Gradients wrt inputs
  if (inputs_bprop != nullptr) {
    k_child->set_output(CreateGradOutputs(k_child, inputs_bprop, f_app, has_aux_, get_value_));
    return;
  }
  // Gradients wrt first input.
  // b_app returns (EnvInstance(grads wrt params), grads wrt input0, grads wrt input1, ...),
  // so obtain first input grad by setting tail_type of Tail to kGradFirst.
  TailPtr tail_grad_first = std::make_shared<Tail>("tail_grad_first", kGradFirst);
  tail_grad_first->set_enable_tuple_grad_first(forward_graph->has_flag("enable_tuple_grad_first"));
  auto tail_grad_first_cnode = k_child->NewCNodeInOrder({NewValueNode(tail_grad_first), b_app});
  k_child->set_output(CreateGradOutputs(k_child, tail_grad_first_cnode, f_app, has_aux_, get_value_));
}

namespace {
// Check if primal func graph has the primitive returned sparse result in its bprop().
void CheckPrimBpropReturnSparse(const FuncGraphPtr &primal_graph) {
  bool has_sparse_bprop_prim = false;
  (void)TopoSort(primal_graph->return_node(), SuccDeeperSimple,
                 [&has_sparse_bprop_prim](const AnfNodePtr &node) -> IncludeType {
                   MS_EXCEPTION_IF_NULL(node);
                   if (has_sparse_bprop_prim) {
                     return EXCLUDE;
                   }
                   PrimitivePtr prim = nullptr;
                   if (node->isa<CNode>()) {
                     prim = GetCNodePrimitiveWithoutDoSignature(node);
                   } else {
                     prim = GetPrimitiveWithoutDoSignature(node);
                   }
                   if (prim != nullptr) {
                     bool sparse_bprop = GetPrimitiveFlag(prim, GRAPH_FLAG_BPROP_RETURN_SPARSE);
                     if (sparse_bprop) {
                       MS_LOG(DEBUG) << "prim: " << prim->ToString() << " has attr 'bprop_return_sparse'";
                       has_sparse_bprop_prim = true;
                       return EXCLUDE;
                     }
                   }
                   return FOLLOW;
                 });
  if (has_sparse_bprop_prim) {
    primal_graph->set_flag(FUNC_GRAPH_FLAG_SPARSE_BPROP, true);
    EnvSetSparseResultMgr::GetInstance().Set(true);
  }
}
}  // namespace

// Generate the graph.
FuncGraphPtr GradOperation::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION)
      << "'GradOperation' requires a forward network or function as an input, while the input is empty.";
  }

  constexpr size_t fn_index = 0;
  auto fn_abs = args_abs_list[fn_index];
  constexpr size_t len_with_weight = 2;
  constexpr size_t weights_index = 1;
  if (args_abs_list.size() >= len_with_weight) {
    weight_value_ = args_abs_list[weights_index];
  }
  MS_EXCEPTION_IF_NULL(fn_abs);
  if (fn_abs->isa<AbstractClass>()) {
    auto class_abs = dyn_cast<AbstractClass>(fn_abs);
    auto class_val = class_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(class_val);
    auto class_obj = class_val->cast<parse::MsClassObjectPtr>();
    MS_EXCEPTION_IF_NULL(class_obj);
    auto obj_name = std::regex_replace(class_obj->name(), std::regex("MsClassObject:"), "");
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell' type "
                      << "object, but got object with jit_class type" << obj_name << ".";
  }
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(fn_abs);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got "
                      << args_abs_list[0]->ToString();
  }

  auto real_fn = fn->cast_ptr<FuncGraphAbstractClosure>();
  if (real_fn == nullptr) {
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got "
                      << fn->ToString();
  }
  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);

  if (has_aux_) {
    GradAuxPtr aux_fn = std::make_shared<GradAux>("aux_fn");
    auto output_cnode = forward_graph->output();
    auto aux_fn_cnode = forward_graph->NewCNodeInOrder({NewValueNode(aux_fn), output_cnode, NewValueNode(get_value_)});
    forward_graph->set_output(aux_fn_cnode);
  }

  forward_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);

  // Check if primal func graph has the primitive returned sparse result in its bprop().
  CheckPrimBpropReturnSparse(forward_graph);

  FuncGraphPtr grad_fg = nullptr;
  {
    TraceGuard g(MakeTraceInfo<TraceGradOperation>(forward_graph->debug_info()));
    grad_fg = std::make_shared<FuncGraph>();
  }
  auto nparam = forward_graph->parameters().size();

  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  grad_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  grad_fg->set_flag("grad_fg", true);
  if (grad_fg->debug_info() != nullptr) {
    grad_fg->debug_info()->set_name(ss.str());
  }
  ParameterPtr param_graph = grad_fg->add_parameter();

  bool is_weights_empty_or_none = false;
  AnfNodePtr weights = nullptr;
  AnfNodePtr position = nullptr;
  if (args_abs_list.size() > weights_index) {
    auto weights_abs = args_abs_list[weights_index];
    MS_EXCEPTION_IF_NULL(weights_abs);
    if (weights_abs->isa<AbstractSequence>()) {
      if (weights_abs->cast<AbstractSequencePtr>()->empty()) {
        is_weights_empty_or_none = true;
      }
    }
  }
  if (get_by_position_) {
    weights = grad_fg->add_parameter();
    position = grad_fg->add_parameter();
  } else if (get_by_list_) {
    weights = grad_fg->add_parameter();
    // Check if weights is None.
    if (!is_weights_empty_or_none && args_abs_list.size() > weights_index) {
      auto weights_abs = args_abs_list[weights_index];
      MS_EXCEPTION_IF_NULL(weights_abs);
      if (weights_abs->isa<AbstractNone>()) {
        is_weights_empty_or_none = true;
      }
    }
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimJ));
  inputs.push_back(param_graph);
  auto j = grad_fg->NewCNodeInOrder(inputs);
  if (merge_forward_) {
    j->set_user_data<bool>("merge_forward", std::make_shared<bool>(true));
  }
  // df is checked in GetGrad
  FuncGraphPtr k_child = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceGradOperation>(forward_graph->debug_info()));
    k_child = GetGrad(j, weights, position, forward_graph, is_weights_empty_or_none);
    k_child->set_flag(FUNC_GRAPH_FLAG_ARGS_NO_EXPAND, true);
  }
  grad_fg->set_output(NewValueNode(k_child));

  return grad_fg;
}

// Generate the vmap_graph.
VmapOperation::VmapOperation(const std::string &name) : MetaFuncGraph(name) {
  auto default_zero = std::make_shared<Int64Imm>(static_cast<int64_t>(0));
  signatures_ =
    // def vmap(func:read, in_axes:ref, out_axes:ref):
    std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                            {"in_axes", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault, default_zero,
                             SignatureEnumDType::kDTypeEmptyDefaultValue},
                            {"out_axes", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault, default_zero,
                             SignatureEnumDType::kDTypeEmptyDefaultValue}});
}

FuncGraphPtr VmapOperation::GetVmap(const AnfNodePtr &vmap, int param_number) const {
  FuncGraphPtr vmap_child = std::make_shared<FuncGraph>();
  vmap_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  vmap_child->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, true);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(vmap);
  for (int i = 0; i < param_number; ++i) {
    inputs.push_back(vmap_child->add_parameter());
  }
  auto vmap_app = vmap_child->NewCNodeInOrder(inputs);
  vmap_child->set_output(vmap_app);

  return vmap_child;
}

namespace {
bool IsAxesAllNone(const ValuePtr &axes) {
  MS_EXCEPTION_IF_NULL(axes);
  ValueSequencePtr axes_seq = dyn_cast<ValueSequence>(axes);
  MS_EXCEPTION_IF_NULL(axes_seq);
  auto axes_seq_value = axes_seq->value();
  if (std::all_of(axes_seq_value.begin(), axes_seq_value.end(), [](const ValuePtr &axes_value_ptr) {
        if (axes_value_ptr->isa<ValueSequence>()) {
          return IsAxesAllNone(axes_value_ptr);
        }
        if (!axes_value_ptr->isa<None>()) {
          return false;
        }
        return true;
      })) {
    return true;
  }
  return false;
}

ValuePtr CheckAxes(const AbstractBasePtr &axes_abs, bool is_in_axes = false, int nparam = 0, size_t cell_size = 0) {
  ValuePtr axes_value = nullptr;
  auto axes_name = is_in_axes ? "in_axes" : "out_axes";

  auto axes_abs_sequence = dyn_cast<AbstractSequence>(axes_abs);
  if (axes_abs_sequence != nullptr) {
    axes_value = axes_abs->cast<AbstractSequencePtr>()->ElementsBuildValue<ValueTuple>();
    MS_EXCEPTION_IF_NULL(axes_value);
    if (is_in_axes) {
      ValueSequencePtr in_axes_seq = dyn_cast<ValueSequence>(axes_value);
      int in_axes_size = SizeToInt(in_axes_seq->size());
      if (nparam != in_axes_size) {
        MS_LOG(EXCEPTION) << "When vmap`s '" << axes_name
                          << "' is a tuple or list, and its size must be equal to the number of arguments of 'fn': "
                          << nparam << ", but got size: " << in_axes_size << ".";
      }
    }
    bool elem_all_none = IsAxesAllNone(axes_value);
    if (elem_all_none && cell_size == 0) {
      MS_LOG(EXCEPTION) << "The '" << axes_name
                        << "' of 'vmap' cannot be all None while 'fn' is not a 'CellList', but got "
                        << axes_value->ToString() << ".";
    }
  } else {
    axes_value = axes_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(axes_value);
    if (axes_value->isa<None>() && cell_size == 0) {
      MS_LOG(EXCEPTION) << "The '" << axes_name
                        << "' of 'vmap' cannot be a single None while 'fn' is not a 'CellList'.";
    } else if (!axes_value->isa<None>() && !axes_value->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "The axis in vmap`s '" << axes_name << "' can only be of type Int or None, but got "
                        << axes_abs->ToString() << ".";
    }
  }
  return axes_value;
}

DebugInfoPtr CheckVmapFunc(const AbstractBasePtr &fn_arg, int *nparam, size_t *cell_size) {
  DebugInfoPtr origin_graph_info = nullptr;
  // In the model ensembling parallel training scenario, fn is a CellList.
  AbstractTuplePtr cell_list = dyn_cast<AbstractTuple>(fn_arg);
  if (cell_list != nullptr) {
    *cell_size = cell_list->size();
    if (*cell_size <= 1) {
      MS_LOG(EXCEPTION) << "In the model ensembling parallel training scenario ('VmapOperation' arg0 is a 'CellList'),"
                        << " the size of 'CellList' must be greater than 1, but got " << *cell_size << ".";
    }
    const AbstractBasePtrList &cell_list_fns = cell_list->elements();
    for (auto fn_abs : cell_list_fns) {
      MS_EXCEPTION_IF_NULL(fn_abs);
      AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(fn_abs);
      if (fn == nullptr) {
        MS_LOG(EXCEPTION) << "'VmapOperation' arg0 is a 'CellList', whose elements must be 'Cell', but got "
                          << fn_abs->ToString() << ".";
      }
      auto partial_fn = dyn_cast<PartialAbstractClosure>(fn_abs);
      if (partial_fn != nullptr) {
        fn = partial_fn->fn();
      }
      auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
      if (real_fn == nullptr) {
        MS_LOG(EXCEPTION) << "'VmapOperation' arg0 is a 'CellList', whose element " << fn->ToString()
                          << " cast to 'FuncGraphAbstractClosure' failed.";
      }

      FuncGraphPtr orig_graph = real_fn->func_graph();
      MS_EXCEPTION_IF_NULL(orig_graph);
      orig_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      int fn_nparam =
        SizeToInt(orig_graph->parameters().size() - (partial_fn != nullptr ? partial_fn->args().size() : 0));
      if (*nparam == -1) {
        origin_graph_info = orig_graph->debug_info();
        *nparam = fn_nparam;
      } else if (*nparam != fn_nparam) {
        MS_LOG(EXCEPTION) << "'VmapOperation' arg0 is a CellList, whose elements's inputs should be consistent.";
      }
    }
  } else {
    AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(fn_arg);
    if (fn == nullptr) {
      MS_LOG(EXCEPTION) << "'VmapOperation' arg0 must be a 'Function' or 'Cell', but got " << fn_arg->ToString() << ".";
    }
    auto partial_fn = dyn_cast<PartialAbstractClosure>(fn);
    if (partial_fn != nullptr) {
      fn = partial_fn->fn();
    }
    auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
    if (real_fn == nullptr) {
      MS_LOG(EXCEPTION) << "'VmapOperation' arg0 " << fn->ToString() << " cast to 'FuncGraphAbstractClosure' failed.";
    }

    FuncGraphPtr orig_graph = real_fn->func_graph();
    MS_EXCEPTION_IF_NULL(orig_graph);
    orig_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    *nparam = SizeToInt(orig_graph->parameters().size() - (partial_fn != nullptr ? partial_fn->args().size() : 0));
    origin_graph_info = orig_graph->debug_info();
  }
  return origin_graph_info;
}
}  // namespace

FuncGraphPtr VmapOperation::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION) << "'VmapOperation' requires a network or function as an input, while the input is empty.";
  }

  constexpr auto vmap_operation_input_num = 3;
  const std::string op_name = "vmap";
  CheckArgsSize(op_name, args_abs_list, vmap_operation_input_num);

  auto fn_arg = args_abs_list[0];
  auto in_axes_arg = args_abs_list[1];
  auto out_axes_arg = args_abs_list[2];

  int nparam = -1;
  size_t cell_size = 0;
  DebugInfoPtr origin_graph_info = CheckVmapFunc(fn_arg, &nparam, &cell_size);

  FuncGraphPtr vmap_fg = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceVmapOperation>(origin_graph_info));
    vmap_fg = std::make_shared<FuncGraph>();
  }

  std::ostringstream ss;
  ss << "vmap{" << nparam << "}";
  vmap_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (vmap_fg->debug_info() != nullptr) {
    vmap_fg->debug_info()->set_name(ss.str());
  }

  // Add parameter for `fn`, `in_axes` and `out_axes` respectively.
  ParameterPtr param_graph = vmap_fg->add_parameter();
  (void)vmap_fg->add_parameter();
  (void)vmap_fg->add_parameter();

  // Validity verification of in_axes and out_axes
  ValuePtr in_axes = CheckAxes(in_axes_arg, true, nparam, cell_size);
  ValuePtr out_axes = CheckAxes(out_axes_arg);

  PrimitivePtr kprim_vmap = std::make_shared<Primitive>(kVmapOpName, kSideEffectPropagate);
  kprim_vmap->set_attr("in_axes", in_axes);
  kprim_vmap->set_attr("out_axes", out_axes);
  kprim_vmap->set_attr("cell_size", MakeValue(cell_size));

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(kprim_vmap));
  inputs.push_back(param_graph);
  auto vmap = vmap_fg->NewCNodeInOrder(inputs);

  FuncGraphPtr vmap_child = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceVmapOperation>(origin_graph_info));
    vmap_child = GetVmap(vmap, nparam);
  }

  vmap_fg->set_output(NewValueNode(vmap_child));
  return vmap_fg;
}

TaylorOperation::TaylorOperation(const std::string &name) : MetaFuncGraph(name) {
  // def Taylor(func:read):
  signatures_ = std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault}});
}

FuncGraphPtr TaylorOperation::GetTaylorGrad(const AnfNodePtr &k,
                                            const std::vector<AnfNodePtr> &forward_graph_params) const {
  FuncGraphPtr k_child = std::make_shared<FuncGraph>();
  k_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(k);
  MS_LOG(INFO) << "TaylorOperation forward input size " << forward_graph_params.size();
  for (size_t i = 0; i < forward_graph_params.size(); ++i) {
    inputs.push_back(k_child->add_parameter());
  }
  // Taylor(fn)(input params)
  auto k_app = k_child->NewCNodeInOrder(inputs);

  k_child->set_output(k_app);
  return k_child;
}

// Generate the graph to calculate higher order derivatives.
FuncGraphPtr TaylorOperation::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION)
      << "'TaylorOperation' requires a forward network or function as an input, while the input is empty.";
  }

  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_abs_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "'TaylorOperation' arg0 must be a 'Function' or 'Cell', but got "
                      << args_abs_list[0]->ToString();
  }

  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);

  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr grad_fg = nullptr;
  MS_LOG(INFO) << "'TaylorOperation' forward_graph" << forward_graph->debug_info();
  grad_fg = std::make_shared<FuncGraph>();
  auto nparam = forward_graph->parameters().size();

  std::ostringstream ss;
  ss << "taylorgrad{" << nparam << "}";
  grad_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (grad_fg->debug_info() != nullptr) {
    grad_fg->debug_info()->set_name(ss.str());
  }
  ParameterPtr param_graph = grad_fg->add_parameter();

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimTaylor));
  inputs.push_back(param_graph);
  // Taylor(fn)
  auto mark_taylor = grad_fg->NewCNodeInOrder(inputs);
  FuncGraphPtr k_child = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceGradOperation>(forward_graph->debug_info()));
    k_child = GetTaylorGrad(mark_taylor, forward_graph->parameters());
  }
  grad_fg->set_output(NewValueNode(k_child));
  // return Taylor(fn)(inputs)
  return grad_fg;
}

FuncGraphPtr TupleAdd::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // args: tuple1, tuple2
  abstract::CheckArgsSize("TupleAdd", args_abs_list, 2);
  AbstractBasePtr abs_a = args_abs_list[0];
  AbstractBasePtr abs_b = args_abs_list[1];

  AbstractTuplePtr a_tuple = dyn_cast<AbstractTuple>(abs_a);
  AbstractTuplePtr b_tuple = dyn_cast<AbstractTuple>(abs_b);
  if (a_tuple == nullptr || b_tuple == nullptr) {
    TypePtrList types;
    (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(types),
                         [](const AbstractBasePtr &arg) -> TypePtr {
                           MS_EXCEPTION_IF_NULL(arg);
                           return arg->BuildType();
                         });
    auto stub = GenerateStubFunc(types);
    if (stub != nullptr) {
      MS_LOG(DEBUG) << "GenerateStubFunc for TupleAdd, function: " << stub->ToString();
      return stub;
    }
    MS_LOG(EXCEPTION) << "The type of argument in TupleAdd operator should be tuple, but the first argument is "
                      << abs_a->ToString() << ", the second argument is " << abs_b->ToString();
  }

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_tup_a = ret->add_parameter();
  AnfNodePtr p_tup_b = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeTuple));

  int64_t tuple_size = SizeToLong(a_tuple->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), p_tup_a, NewValueNode(i)}));
  }

  tuple_size = SizeToLong(b_tuple->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), p_tup_b, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNodeInOrder(elems));
  return ret;
}

FuncGraphPtr ListAdd::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // args: list1, list2
  abstract::CheckArgsSize("ListAdd", args_abs_list, 2);
  AbstractBasePtr abs_a = args_abs_list[0];
  AbstractBasePtr abs_b = args_abs_list[1];

  AbstractListPtr a_list = dyn_cast<AbstractList>(abs_a);
  AbstractListPtr b_list = dyn_cast<AbstractList>(abs_b);
  if (a_list == nullptr || b_list == nullptr) {
    TypePtrList types;
    (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(types),
                         [](const AbstractBasePtr &arg) -> TypePtr {
                           MS_EXCEPTION_IF_NULL(arg);
                           return arg->BuildType();
                         });
    auto stub = GenerateStubFunc(types);
    if (stub != nullptr) {
      MS_LOG(DEBUG) << "GenerateStubFunc for ListAdd, function: " << stub->ToString();
      return stub;
    }
    MS_LOG(EXCEPTION) << "The type of argument in ListAdd operator should be list, but the first argument is "
                      << abs_a->ToString() << ", the second argument is " << abs_b->ToString();
  }

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_list_a = ret->add_parameter();
  AnfNodePtr p_list_b = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));

  int64_t tuple_size = SizeToLong(a_list->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), p_list_a, NewValueNode(i)}));
  }

  tuple_size = SizeToLong(b_list->size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    elems.push_back(ret->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), p_list_b, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNodeInOrder(elems));
  return ret;
}

int64_t GetArgScalarValue(const abstract::AbstractScalarPtr &scalar, const std::string &) {
  MS_EXCEPTION_IF_NULL(scalar);
  return GetValue<int64_t>(scalar->BuildValue());
}

int64_t GetPositiveIndex(int64_t index, int64_t length) {
  if (index < 0) {
    index += length;
  }
  return index;
}

int64_t CheckSliceMember(const AbstractBasePtr &member, int64_t default_value, const std::string &member_name) {
  MS_EXCEPTION_IF_NULL(member);

  if (member->isa<AbstractScalar>()) {
    return GetArgScalarValue(dyn_cast<AbstractScalar>(member), member_name);
  }

  if (member->isa<AbstractNone>()) {
    return default_value;
  }

  if (member->isa<AbstractTensor>()) {
    MS_EXCEPTION(TypeError)
      << "The argument of SliceMember operator must be a Scalar or None or constant Tensor, but got a variable Tensor";
  }
  MS_EXCEPTION_IF_NULL(member->BuildType());
  MS_EXCEPTION(TypeError)
    << "The argument of SliceMember operator must be a Scalar or None or constant Tensor, but got "
    << member->BuildType()->ToString();
}

std::tuple<int64_t, int64_t, int64_t> GenerateTupleSliceParameter(const AbstractSequencePtr &sequence,
                                                                  const AbstractSlicePtr &slice) {
  MS_EXCEPTION_IF_NULL(sequence);
  MS_EXCEPTION_IF_NULL(slice);
  int64_t start_index;
  int64_t stop_index;
  int64_t step_value;

  const std::string start_name("Slice start index");
  const std::string stop_name("Slice stop index");
  const std::string step_name("Slice step value");

  int64_t tuple_size = SizeToLong(sequence->size());
  int64_t start_default = 0;
  int64_t stop_default = tuple_size;
  int64_t step_default = kStepDefault;

  step_value = CheckSliceMember(slice->step(), step_default, step_name);
  if (step_value == 0) {
    MS_EXCEPTION(ValueError) << "Slice step cannot be zero.";
  }

  if (step_value < 0) {
    start_default = tuple_size - 1;
    stop_default = ((-tuple_size) - 1);
  }

  start_index = CheckSliceMember(slice->start(), start_default, start_name);
  stop_index = CheckSliceMember(slice->stop(), stop_default, stop_name);

  if (start_index < -tuple_size) {
    start_index = 0;
  }

  if (stop_index > tuple_size) {
    stop_index = tuple_size;
  }

  if (start_index > tuple_size) {
    start_index = tuple_size;
  }

  if (stop_index < ((-tuple_size) - 1)) {
    stop_index = 0;
  }

  start_index = GetPositiveIndex(start_index, tuple_size);

  stop_index = GetPositiveIndex(stop_index, tuple_size);

  return std::make_tuple(start_index, stop_index, step_value);
}

void SequenceSliceGetItem::CheckArgs(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t arg_size = 2;
  abstract::CheckArgsSize(this->name(), args_abs_list, arg_size);
  sequence_ = abstract::CheckArg<AbstractSequence>(this->name(), args_abs_list, 0);
  slice_ = abstract::CheckArg<AbstractSlice>(this->name(), args_abs_list, 1);
}

FuncGraphPtr SequenceSliceGetItem::BuildFuncGraph(int64_t start_index, int64_t stop_index, int64_t step_value) {
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  AnfNodePtr p_seq = ret->add_parameter();
  (void)ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim_));
  if (step_value > 0) {
    for (int64_t index = start_index; index < stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNodeInOrder({NewValueNode(get_item_), p_seq, NewValueNode(index)}));
    }
  } else {
    for (int64_t index = start_index; index > stop_index; index = index + step_value) {
      elems.push_back(ret->NewCNodeInOrder({NewValueNode(get_item_), p_seq, NewValueNode(index)}));
    }
  }

  ret->set_output(ret->NewCNodeInOrder(elems));
  return ret;
}

FuncGraphPtr TupleGetItemTensor::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // select indexed item
  // args: tuple of items, index
  const std::string op_name = std::string("TupleGetItemTensor");
  const size_t inputs_size = 2;
  abstract::CheckArgsSize(op_name, args_abs_list, inputs_size);
  auto ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto tuple = ret_graph->add_parameter();
  auto index = ret_graph->add_parameter();

  constexpr size_t tuple_index = 0;
  auto abs = args_abs_list[tuple_index];
  MS_EXCEPTION_IF_NULL(abs);
  auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if (!tuple_abs->dynamic_len()) {
    const auto &elements = tuple_abs->elements();
    if (std::all_of(elements.begin(), elements.end(), [](const AbstractBasePtr &e) {
          MS_EXCEPTION_IF_NULL(e);
          return e->isa<abstract::FuncGraphAbstractClosure>() || e->isa<abstract::PartialAbstractClosure>() ||
                 e->isa<abstract::PrimitiveAbstractClosure>();
        })) {
      ret_graph->set_output(ret_graph->NewCNodeInOrder({NewValueNode(prim::kPrimSwitchLayer), index, tuple}));
      return ret_graph;
    }
  }

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    MS_EXCEPTION(TypeError) << "When JIT_SYNTAX_LEVEL is STRICT, using Tensor index to get value from tuple requires "
                            << "that all elements in tuple should be function but got tuple abstract: "
                            << tuple_abs->ToString();
  }
  // Script
  constexpr auto internal_tuple_input = "__internal_tuple_input__";
  constexpr auto internal_index_input = "__internal_index_input__";
  std::stringstream script_buffer;
  script_buffer << internal_tuple_input << "[" << internal_index_input << "]";
  const std::string &script = script_buffer.str();
  const auto script_str = std::make_shared<StringImm>(script);
  // Key
  std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)key_value_names_list.emplace_back(NewValueNode(internal_tuple_input));
  (void)key_value_names_list.emplace_back(NewValueNode(internal_index_input));
  const auto key_value_name_tuple = ret_graph->NewCNode(key_value_names_list);
  // Value
  std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)key_value_list.emplace_back(tuple);
  (void)key_value_list.emplace_back(index);
  const auto key_value_tuple = ret_graph->NewCNode(key_value_list);
  auto res =
    fallback::CreatePyExecuteCNode(ret_graph, NewValueNode(script_str), key_value_name_tuple, key_value_tuple, nullptr);
  ret_graph->set_output(res);
  return ret_graph;
}

namespace {
FuncGraphPtr GetShard(const AnfNodePtr &shard, const std::vector<AnfNodePtr> &origin_graph_params) {
  FuncGraphPtr shard_child = std::make_shared<FuncGraph>();
  shard_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  std::vector<AnfNodePtr> inputs;
  inputs.reserve(origin_graph_params.size() + 1);
  (void)inputs.emplace_back(shard);
  for (size_t i = 0; i < origin_graph_params.size(); ++i) {
    (void)inputs.emplace_back(shard_child->add_parameter());
  }
  auto shard_app = shard_child->NewCNodeInOrder(std::move(inputs));

  shard_child->set_output(shard_app);
  return shard_child;
}
}  // namespace

FuncGraphPtr Shard::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.size() != kShardInputSize) {
    MS_LOG(EXCEPTION) << "'Shard' requires " << kShardInputSize
                      << " inputs. Includes a Cell or function, in_axes, out_axes, parameter_plan, device and level.";
  }

  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  AbstractFunctionPtr fn = dyn_cast<AbstractFunction>(args_abs_list[0]);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "'Shard' arg0 must be a 'Function' or 'Cell', but got " << args_abs_list[0]->ToString() << ".";
  }

  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(fn);
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr origin_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(origin_graph);
  origin_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  FuncGraphPtr shard_fg = nullptr;
  {
    TraceGuard g(MakeTraceInfo<TraceShard>(origin_graph->debug_info()));
    shard_fg = std::make_shared<FuncGraph>();
  }
  // Create the debug info
  auto parameter_size = origin_graph->parameters().size();
  std::ostringstream ss;
  ss << "shard{" << parameter_size << "}";
  shard_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (shard_fg->debug_info() != nullptr) {
    shard_fg->debug_info()->set_name(ss.str());
  }
  // Make the Shard node.
  std::vector<AnfNodePtr> inputs;
  inputs.reserve(args_abs_list.size() + 1);
  (void)inputs.emplace_back(NewValueNode(prim::kPrimShard));
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    (void)inputs.emplace_back(shard_fg->add_parameter());
  }
  auto shard = shard_fg->NewCNodeInOrder(std::move(inputs));

  FuncGraphPtr shard_child = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceShard>(shard_fg->debug_info()));
    shard_child = GetShard(shard, origin_graph->parameters());
  }
  shard_fg->set_output(NewValueNode(shard_child));
  return shard_fg;
}

FuncGraphPtr AddAttr::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (kAddAttrInputSize != args_abs_list.size()) {
    MS_LOG(EXCEPTION) << "'AddAttr' requires " << kAddAttrInputSize << " inputs. Includes func, attr_dict.";
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  AbstractFunctionPtr abs_func = dyn_cast<AbstractFunction>(args_abs_list[0]);
  if (!abs_func) {
    MS_LOG(EXCEPTION) << "'AddAttr' 0-th arg must be a 'Function', but got " << args_abs_list[0]->ToString();
  }
  auto real_fn = dyn_cast<FuncGraphAbstractClosure>(abs_func);
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr input_fg = real_fn->func_graph();
  input_fg->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  MS_EXCEPTION_IF_NULL(input_fg);
  FuncGraphPtr addattr_fg = nullptr;
  {
    TraceGuard g(MakeTraceInfo<TraceAddAttr>(input_fg->debug_info()));
    addattr_fg = std::make_shared<FuncGraph>();
  }
  addattr_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  // Create the debug info
  auto parameter_size = input_fg->parameters().size();
  std::ostringstream oss;
  oss << "add_attr{" << parameter_size << "}";
  addattr_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (addattr_fg->debug_info() != nullptr) {
    addattr_fg->debug_info()->set_name(oss.str());
  }
  // Make AddAttr Node
  AnfNodePtrList addattr_inputs;
  (void)addattr_inputs.emplace_back(NewValueNode(prim::kPrimAddAttr));
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    (void)addattr_inputs.emplace_back(addattr_fg->add_parameter());
  }
  CNodePtr addattr_node = addattr_fg->NewCNodeInOrder(std::move(addattr_inputs));
  addattr_fg->set_output(addattr_node);
  return addattr_fg;
}

void ListSliceSetItem::CheckArgs(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t kSliceSetItemArgsSizeargs_size = 3;
  constexpr size_t kSliceSetItemListIndex = 0;
  constexpr size_t kSliceSetItemSliceIndex = 1;
  constexpr size_t kSliceSetItemValueIndex = 2;
  abstract::CheckArgsSize("list_slice_set_item", args_abs_list, kSliceSetItemArgsSizeargs_size);
  this->sequence_ = abstract::CheckArg<AbstractList>("list_slice_set_item", args_abs_list, kSliceSetItemListIndex);
  this->slice_ = abstract::CheckArg<AbstractSlice>("list_slice_set_item", args_abs_list, kSliceSetItemSliceIndex);
  this->value_list_ = abstract::CheckArg<AbstractList>("list_slice_set_item", args_abs_list, kSliceSetItemValueIndex);
}

FuncGraphPtr ListSliceSetItem::BuildFuncGraph(int64_t start_index, int64_t stop_index, int64_t step_value) {
  // Init graph with the input list_node slice assign_node
  CheckAssignRange(start_index, stop_index, step_value);
  auto graph = std::make_shared<FuncGraph>();
  graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto list_node = graph->add_parameter();
  (void)graph->add_parameter();
  auto assign_parameter = graph->add_parameter();
  auto assign_node = GetAssignNode(graph, assign_parameter, step_value);
  std::vector<AnfNodePtr> elems = {NewValueNode(prim::kPrimMakeList)};
  int64_t list_index = 0;
  // check the index is in the slice range
  auto check_in_range = [start_index, stop_index, step_value](int64_t index) -> bool {
    if (step_value > 0) {
      return (index >= start_index && index < stop_index);
    }
    return (index <= start_index && index > stop_index);
  };
  int64_t list_size = SizeToLong(sequence_->size());
  int64_t assign_index = 0;
  int64_t value_size = SizeToLong(value_list_->size());
  while (list_index < list_size || assign_index < value_size) {
    if (!check_in_range(list_index)) {
      // list start <= stop && step = 1 insert the assign node to target node
      while (assign_index < value_size && list_index == start_index) {
        (void)elems.emplace_back(
          graph->NewCNodeInOrder({NewValueNode(kPrimListGetItem), assign_node, NewValueNode(assign_index++)}));
      }
      if (list_index < list_size) {
        (void)elems.emplace_back(
          graph->NewCNodeInOrder({NewValueNode(kPrimListGetItem), list_node, NewValueNode(list_index++)}));
      }
    } else {
      if (((list_index - start_index) % step_value) == 0) {
        ++list_index;
        if (assign_index >= value_size) {
          continue;
        }
        (void)elems.emplace_back(
          graph->NewCNodeInOrder({NewValueNode(kPrimListGetItem), assign_node, NewValueNode(assign_index++)}));
      } else {
        (void)elems.emplace_back(
          graph->NewCNodeInOrder({NewValueNode(kPrimListGetItem), list_node, NewValueNode(list_index++)}));
      }
      // the assign node's len is larger than the range
      while (!check_in_range(list_index) && assign_index < value_size) {
        (void)elems.emplace_back(
          graph->NewCNodeInOrder({NewValueNode(kPrimListGetItem), assign_node, NewValueNode(assign_index++)}));
      }
    }
  }

  graph->set_output(graph->NewCNodeInOrder(elems));
  return graph;
}

void ListSliceSetItem::CheckAssignRange(int64_t start_index, int64_t stop_index, int64_t step_value) {
  if (step_value != kStepDefault) {
    auto range = stop_index - start_index;
    int include_start = (range % step_value) == 0 ? 0 : 1;
    auto assign_size = (range / step_value) + include_start;
    assign_size = assign_size > 0 ? assign_size : 0;
    if (assign_size != SizeToLong(value_list_->size())) {
      MS_EXCEPTION(ValueError) << "attempt to assign sequence of size " << value_list_->size()
                               << " to extended slice of size " << assign_size;
    }
  }
}

AnfNodePtr ListSliceSetItem::GetAssignNode(const FuncGraphPtr &func_graph, const AnfNodePtr &assign_node,
                                           int64_t step_value) {
  if (step_value > 0) {
    return assign_node;
  }
  std::vector<AnfNodePtr> elems = {NewValueNode(prim::kPrimMakeList)};
  for (int64_t i = SizeToInt(value_list_->size()) - 1; i >= 0; --i) {
    (void)elems.emplace_back(
      func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimListGetItem), assign_node, NewValueNode(i)}));
  }
  return func_graph->NewCNodeInOrder(elems);
}

FuncGraphPtr SequenceSlice::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  this->CheckArgs(args_abs_list);
  auto [start, stop, step] = GenerateTupleSliceParameter(sequence_, slice_);
  return this->BuildFuncGraph(start, stop, step);
}

FuncGraphPtr ZerosLike::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr auto input_size = 1;
  abstract::CheckArgsSize("ZerosLike", args_abs_list, input_size);

  auto x = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(x);
  auto type = x->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() == kTuple->type_id() || type->type_id() == kList->type_id()) {
    auto abs_seq = x->cast<AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->dynamic_len()) {
      FuncGraphPtr res_graph = std::make_shared<FuncGraph>();
      res_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
      if (res_graph->debug_info() != nullptr) {
        res_graph->debug_info()->set_name("zeros_like");
      }
      auto x_parameter = res_graph->add_parameter();
      res_graph->set_output(res_graph->NewCNodeInOrder({NewValueNode(prim::kPrimSequenceZerosLike), x_parameter}));
      return res_graph;
    }
  }

  HyperMap hyper_map(false, fn_leaf_);
  TypePtrList types;
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(types),
                       [](const AbstractBasePtr &arg) -> TypePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->BuildType();
                       });
  return hyper_map.GenerateFromTypes(types);
}

// IterConvert is used when the input is need to convert to Iterable object.
FuncGraphPtr IterConverter::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr auto input_size = 1;
  abstract::CheckArgsSize("IterConverter", args_abs_list, input_size);
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  if (input_abs->isa<abstract::AbstractAny>() || input_abs->BuildValue()->isa<parse::InterpretedObject>()) {
    const std::vector<std::string> funcs_str{"tuple"};
    auto ret_node = fallback::GeneratePyInterpretWithAbstract(fg, funcs_str, input_size);
    fg->set_output(ret_node);
    return fg;
  }

  auto input_type = input_abs->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto type_id = input_type->type_id();
  std::vector<int64_t> iterable_valid_types{
    TypeId::kObjectTypeString,     TypeId::kObjectTypeTuple,    TypeId::kObjectTypeList,  TypeId::kObjectTypeDictionary,
    TypeId::kObjectTypeTensorType, TypeId::kObjectTypeFunction, TypeId::kMetaTypeExternal};
  bool iterable = std::any_of(iterable_valid_types.begin(), iterable_valid_types.end(),
                              [type_id](int64_t valid_type) { return valid_type == type_id; });
  if (!iterable) {
    MS_EXCEPTION(TypeError) << "'" << TypeIdToString(type_id, true) << "' object is not iterable";
  }

  auto input = fg->add_parameter();
  if (input_abs->isa<AbstractDictionary>()) {
    auto ret_node = fg->NewCNode({NewValueNode(prim::kPrimDictGetKeys), input});
    fg->set_output(ret_node);
    return fg;
  }
  fg->set_output(input);
  return fg;
}

AnfNodePtr ConvertPyInterpret(const FuncGraphPtr &fg, const AnfNodePtr &input, const std::string &sequence_func_type) {
  AnfNodePtrList local_key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList local_value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  std::stringstream script_buffer;
  script_buffer << sequence_func_type << "(";
  const std::string data_str = "__data__";
  script_buffer << data_str << ")";
  (void)local_key_inputs.emplace_back(NewValueNode(data_str));
  (void)local_value_inputs.emplace_back(input);
  const auto &script = script_buffer.str();
  auto local_key_node = fg->NewCNode(local_key_inputs);
  auto local_value_node = fg->NewCNode(local_value_inputs);
  auto local_dict_node = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  return fallback::CreatePyInterpretCNode(fg, script, py::dict(), local_dict_node);
}

// HasNext is used to check whether the input has next element input.
FuncGraphPtr HasNext::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr auto input_size = 1;
  abstract::CheckArgsSize("HasNext", args_abs_list, input_size);
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  auto input = fg->add_parameter();
  if (input_abs->isa<abstract::AbstractAny>() || input_abs->BuildValue()->isa<parse::InterpretedObject>()) {
    const std::string has_next_func = "__import__('mindspore').common._utils._jit_fallback_has_next_func";
    auto ret = ConvertPyInterpret(fg, input, has_next_func);
    fg->set_output(ret);
    return fg;
  }
  const std::string module = "mindspore._extends.parse.standard_method";
  const std::string func_name = "ms_hasnext";
  py::function fn = python_adapter::GetPyFn(module, func_name);
  auto prim_func = parse::ParsePythonCode(fn);
  auto ret = fg->NewCNode({NewValueNode(prim_func), input});
  fg->set_output(ret);
  return fg;
}

// HasNext is used to check whether the input has next element input.
FuncGraphPtr Next::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr auto input_size = 1;
  abstract::CheckArgsSize("Next", args_abs_list, input_size);
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  auto input = fg->add_parameter();
  if (input_abs->isa<abstract::AbstractAny>() || input_abs->BuildValue()->isa<parse::InterpretedObject>()) {
    const std::string next_func = "__import__('mindspore').common._utils._jit_fallback_next_func";
    auto ret = ConvertPyInterpret(fg, input, next_func);
    fg->set_output(ret);
    return fg;
  }
  const std::string module = "mindspore._extends.parse.standard_method";
  const std::string func_name = input_abs->isa<abstract::AbstractDictionary>() ? "dict_next" : "ms_next";
  py::function fn = python_adapter::GetPyFn(module, func_name);
  auto prim_func = parse::ParsePythonCode(fn);
  auto ret = fg->NewCNode({NewValueNode(prim_func), input});
  fg->set_output(ret);
  return fg;
}

FuncGraphPtr TupleFunc::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.size() > 1) {
    MS_LOG(EXCEPTION) << "For 'TupleFunc', the number of input should be 0 or 1, but got " << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (args_abs_list.size() == 0) {
    auto ret = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple)});
    fg->set_output(ret);
    return fg;
  }

  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  auto input = fg->add_parameter();
  if (fallback::ContainsSequenceAnyType(input_abs)) {
    auto ret = ConvertPyInterpret(fg, input, "tuple");
    fg->set_output(ret);
    return fg;
  } else if (input_abs->isa<abstract::AbstractTuple>()) {
    fg->set_output(input);
    return fg;
  } else if (input_abs->isa<abstract::AbstractList>()) {
    // list to tuple
    if (fallback::SequenceAllElementsIsScalar(input_abs)) {
      auto prim = std::make_shared<Primitive>("ListToTuple");
      auto list_to_tuple = fg->NewCNode({NewValueNode(prim), input});
      fg->set_output(list_to_tuple);
      return fg;
    }
  }
  const std::string module = "mindspore._extends.parse.standard_method";
  const std::string func_name = "tuple_func";
  py::function fn = python_adapter::GetPyFn(module, func_name);
  auto prim_func = parse::ParsePythonCode(fn);
  auto ret = fg->NewCNode({NewValueNode(prim_func), input});
  fg->set_output(ret);
  return fg;
}

FuncGraphPtr ListFunc::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.size() > 1) {
    MS_LOG(EXCEPTION) << "For 'ListFunc', the number of input should be 0 or 1, but got " << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (args_abs_list.size() == 0) {
    auto ret = fg->NewCNode({NewValueNode(prim::kPrimMakeList)});
    fg->set_output(ret);
    return fg;
  }

  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  auto input = fg->add_parameter();
  if (fallback::ContainsSequenceAnyType(input_abs)) {
    auto ret = ConvertPyInterpret(fg, input, "list");
    fg->set_output(ret);
    return fg;
  } else if (input_abs->isa<abstract::AbstractList>()) {
    fg->set_output(input);
    return fg;
  } else if (input_abs->isa<abstract::AbstractTuple>()) {
    // tuple to list
    if (fallback::SequenceAllElementsIsScalar(input_abs)) {
      auto prim = std::make_shared<Primitive>("TupleToList");
      auto tuple_to_list = fg->NewCNode({NewValueNode(prim), input});
      fg->set_output(tuple_to_list);
      return fg;
    }
  }
  const std::string module = "mindspore._extends.parse.standard_method";
  const std::string func_name = "list_func";
  py::function fn = python_adapter::GetPyFn(module, func_name);
  auto prim_func = parse::ParsePythonCode(fn);
  auto ret = fg->NewCNode({NewValueNode(prim_func), input});
  fg->set_output(ret);
  return fg;
}

FuncGraphPtr DictFunc::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // dict has three constructors: dict(**kwargs), dict(mapping, **kwargs), dict(iterable, **kwargs)
  // However, kwargs are not currently supported, so the number of input args is limited to either 0 or 1.
  // Refer to: https://docs.python.org/3/library/stdtypes.html#dict
  if (args_abs_list.size() > 1) {
    MS_LOG(EXCEPTION) << "For 'DictFunc', the number of input should be 0 or 1, but got " << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (args_abs_list.empty()) {
    std::vector<AnfNodePtr> keys{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> values{NewValueNode(prim::kPrimMakeTuple)};
    auto ret = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), fg->NewCNode(keys), fg->NewCNode(values)});
    fg->set_output(ret);
    return fg;
  }

  const AbstractBasePtr &input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  ParameterPtr input = fg->add_parameter();
  if (input_abs->isa<abstract::AbstractDictionary>()) {
    fg->set_output(input);
    return fg;
  }
  const std::string module = "mindspore._extends.parse.standard_method";
  const std::string func_name = "dict_func";
  py::function fn = python_adapter::GetPyFn(module, func_name);
  FuncGraphPtr prim_func = parse::ParsePythonCode(fn);
  auto ret = fg->NewCNodeInOrder({NewValueNode(prim_func), input});
  fg->set_output(ret);
  return fg;
}

FuncGraphPtr ForHalfUnrollLess::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t less_inputs_size = 2;
  if (args_abs_list.size() != less_inputs_size) {
    MS_LOG(EXCEPTION) << "For 'ForHalfUnrollLess', the number of input should be " << less_inputs_size << ", but got "
                      << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name("for_half_unroll_less");
  auto x = fg->add_parameter();
  auto y = fg->add_parameter();
  auto other_value = args_abs_list[1]->BuildValue();
  if (!other_value->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "The other value of ForHalfUnrollLess should be an int64 number, but got "
                            << args_abs_list[1]->BuildType();
  }
  auto other = GetValue<int64_t>(other_value);
  if (other == 0) {
    fg->set_output(NewValueNode(false));
    return fg;
  }
  constexpr auto less_module_name = "mindspore.ops.composite.multitype_ops.less_impl";
  ValuePtr less_op = prim::GetPythonOps("less", less_module_name);
  auto cond_node = fg->NewCNodeInOrder({NewValueNode(less_op), x, y});
  fg->set_output(cond_node);
  return fg;
}

// Check if dout_tuple is (dout, (dout_mask, ops_type));
bool IsDoutTupleContainsMask(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  constexpr size_t expected_size = 2;
  auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
  if (tuple_abs == nullptr || tuple_abs->dynamic_len() || tuple_abs->elements().size() != expected_size) {
    return false;
  }
  constexpr auto index_first = 1;
  const auto &mask_abs = tuple_abs->elements().at(index_first);
  auto mask_tuple_abs = mask_abs->cast<abstract::AbstractTuplePtr>();
  if (mask_tuple_abs == nullptr || mask_tuple_abs->dynamic_len() ||
      mask_tuple_abs->elements().size() != expected_size) {
    return false;
  }
  const auto &dmask_abs = mask_tuple_abs->elements().at(0);
  const auto &ops_type_abs = mask_tuple_abs->elements().at(index_first);
  // If Dout_tuple: (dout, (dout_mask, ops_type), (dout, (dout_mask, ops_type)))
  if (!ops_type_abs->isa<AbstractScalar>() || dmask_abs->isa<AbstractTuple>()) {
    return false;
  }
  auto ops_type_scalar_abs = ops_type_abs->cast<abstract::AbstractScalarPtr>();
  auto ops_type_value = ops_type_scalar_abs->BuildValue();
  if (ops_type_value->ContainsValueAny()) {
    return true;
  }
  if (!ops_type_value->isa<Int64Imm>()) {
    return false;
  }
  auto value = GetValue<int64_t>(ops_type_value);
  return value == OpsType::Type_Normal || value == OpsType::Type_View || value == OpsType::Type_Inplace ||
         value == OpsType::Type_Variable;
}

FuncGraphPtr AccumulateDout::BuildAddOutputFG(const std::string &name, const AbstractBasePtrList &args_abs_list) {
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(name);
  auto dout_tuple_input = fg->add_parameter();
  auto factor_tuple_input = fg->add_parameter();
  auto dout_tuple_abs = args_abs_list[0];
  auto factor_tuple_abs = args_abs_list[1];
  auto dout =
    fg->NewCNodeInOrder({NewValueNode(std::make_shared<prim::GetRealBpropOut>("get_real_dout")), dout_tuple_input});
  auto factor =
    fg->NewCNodeInOrder({NewValueNode(std::make_shared<prim::GetRealBpropOut>("get_real_dout")), factor_tuple_input});
  auto cal_dout = fg->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("hyper_add")), dout, factor});
  if (IsDoutTupleContainsMask(dout_tuple_abs) && IsDoutTupleContainsMask(factor_tuple_abs)) {
    auto mask_tuple =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_tuple_input, NewValueNode(int64_t(1))});
    auto cal_res = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), cal_dout, mask_tuple});
    fg->set_output(cal_res);
    return fg;
  }
  auto generate_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("generate_dout_tuple");
  auto cal_res = fg->NewCNodeInOrder({NewValueNode(generate_dout_tuple), cal_dout});
  if (IsDoutTupleContainsMask(dout_tuple_abs) || IsDoutTupleContainsMask(factor_tuple_abs)) {
    fg->set_output(cal_res);
    return fg;
  }
  auto dout_seq_abs = dyn_cast<AbstractSequence>(dout_tuple_abs);
  MS_EXCEPTION_IF_NULL(dout_seq_abs);
  auto factor_seq_abs = dyn_cast<AbstractSequence>(factor_tuple_abs);
  MS_EXCEPTION_IF_NULL(factor_seq_abs);
  if (dout_seq_abs->elements().size() != factor_seq_abs->elements().size()) {
    fg->set_output(cal_res);
    return fg;
  }
  AnfNodePtrList res_tuple{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < dout_seq_abs->elements().size(); ++i) {
    auto dout_abs = dout_seq_abs->elements()[i];
    auto factor_abs = factor_seq_abs->elements()[i];
    auto dout_node_i =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_tuple_input, NewValueNode(int64_t(i))});
    auto factor_node_i =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(i))});
    if (IsDoutTupleContainsMask(dout_abs) && IsDoutTupleContainsMask(factor_abs)) {
      auto accumulate_dout = std::make_shared<prim::AccumulateDout>("_accumulate_dout");
      auto res = fg->NewCNodeInOrder({NewValueNode(accumulate_dout), dout_node_i, factor_node_i});
      (void)res_tuple.emplace_back(res);
    } else {
      auto dout_i =
        fg->NewCNodeInOrder({NewValueNode(std::make_shared<prim::GetRealBpropOut>("get_real_dout")), dout_node_i});
      auto factor_i =
        fg->NewCNodeInOrder({NewValueNode(std::make_shared<prim::GetRealBpropOut>("get_real_dout")), factor_node_i});
      auto cal_dout_i = fg->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("hyper_add")), dout_i, factor_i});
      auto res = fg->NewCNodeInOrder(
        {NewValueNode(std::make_shared<prim::GenerateBpropOutTuple>("generate_dout_tuple")), cal_dout_i});
      (void)res_tuple.emplace_back(res);
    }
  }
  fg->set_output(fg->NewCNodeInOrder(res_tuple));
  return fg;
}

FuncGraphPtr AccumulateDout::BuildAccumulateInplaceOutputFG(const std::string &name) {
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(name);
  auto dout_tuple_input = fg->add_parameter();
  auto factor_tuple_input = fg->add_parameter();
  auto dout = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_tuple_input, NewValueNode(int64_t(0))});
  auto factor =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(0))});
  auto factor_inner_tuple =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(1))});
  auto dout_inner_tuple =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_tuple_input, NewValueNode(int64_t(1))});
  auto dout_mask =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_inner_tuple, NewValueNode(int64_t(0))});
  auto factor_mask =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_inner_tuple, NewValueNode(int64_t(0))});
  auto dout1 = fg->NewCNodeInOrder(
    {NewValueNode(prim::kPrimSelect), dout_mask, dout, fg->NewCNodeInOrder({NewValueNode(prim::kPrimOnesLike), dout})});
  auto factor1 = fg->NewCNodeInOrder({NewValueNode(prim::kPrimSelect), factor_mask, factor,
                                      fg->NewCNodeInOrder({NewValueNode(prim::kPrimOnesLike), factor})});
  auto cal_dout = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMul), dout1, factor1});
  auto cal_mask = fg->NewCNodeInOrder({NewValueNode(prim::kPrimLogicalOr), dout_mask, factor_mask});
  auto cal_res = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), cal_dout,
                                      fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), cal_mask,
                                                           NewValueNode(int64_t(OpsType::Type_Inplace))})});
  fg->set_output(cal_res);
  return fg;
}

FuncGraphPtr AccumulateDout::BuildSelectOutputFG(const std::string &name) {
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(name);
  auto dout_tuple_input = fg->add_parameter();
  auto factor_tuple_input = fg->add_parameter();
  auto dout = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_tuple_input, NewValueNode(int64_t(0))});
  auto factor =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(0))});
  auto factor_inner_tuple =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(1))});
  auto factor_mask =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_inner_tuple, NewValueNode(int64_t(0))});
  auto factor_temp = fg->NewCNodeInOrder({NewValueNode(prim::kPrimSelect), factor_mask, factor,
                                          fg->NewCNodeInOrder({NewValueNode(prim::kPrimOnesLike), factor})});
  auto cal_dout = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMul), factor_temp, dout});
  auto cal_res = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), cal_dout, factor_inner_tuple});
  fg->set_output(cal_res);
  return fg;
}

FuncGraphPtr AccumulateDout::BuildChooseOutputFG(const std::string &name) {
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name(name);
  auto dout_tuple_input = fg->add_parameter();
  auto factor_tuple_input = fg->add_parameter();
  if (types_["dout"] == OpsType::Type_Variable) {
    fg->set_output(factor_tuple_input);
  } else {
    fg->set_output(dout_tuple_input);
  }
  return fg;
}

// AccumulateDout has two inputs, indicate two dout to accumulate:
//  1) dout_tuple: (dout, (dout_mask, dout_type))
//  2) factor_tuple: (factor, (factor_mask, factor_type))
FuncGraphPtr AccumulateDout::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  CheckAccumulateDoutInputAbstract(args_abs_list);
  if (IsBuildSwitchNode()) {
    auto fg = std::make_shared<FuncGraph>();
    fg->debug_info()->set_name("accumulate_dout");
    auto dout_tuple_input = fg->add_parameter();
    auto factor_tuple_input = fg->add_parameter();
    auto factor_inner_tuple =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_tuple_input, NewValueNode(int64_t(1))});
    auto factor_type =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), factor_inner_tuple, NewValueNode(int64_t(1))});
    auto add_out_cond = fg->NewCNodeInOrder(
      {NewValueNode(prim::GetPythonOps("equal")), factor_type, NewValueNode(int64_t(OpsType::Type_Normal))});
    auto add_output_fg = BuildAddOutputFG("accumulate_dout_add_output", args_abs_list);
    auto inner_fg = std::make_shared<FuncGraph>();
    inner_fg->debug_info()->set_name("accumulate_dout_not_add_dout");
    auto inner_dout_tuple_input = inner_fg->add_parameter();
    auto inner_factor_tuple_input = inner_fg->add_parameter();
    auto inner_factor_inner_tuple = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), inner_factor_tuple_input, NewValueNode(int64_t(1))});
    auto inner_dout_inner_tuple = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), inner_dout_tuple_input, NewValueNode(int64_t(1))});
    auto inner_dout_type = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), inner_dout_inner_tuple, NewValueNode(int64_t(1))});
    auto inner_factor_type = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), inner_factor_inner_tuple, NewValueNode(int64_t(1))});
    auto inner_dout_type_is_inplace = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::GetPythonOps("equal")), inner_dout_type, NewValueNode(int64_t(OpsType::Type_Inplace))});
    auto inner_factor_type_is_inplace = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::GetPythonOps("equal")), inner_factor_type, NewValueNode(int64_t(OpsType::Type_Inplace))});
    auto inner_cond = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::GetPythonOps("logical_and")), inner_dout_type_is_inplace, inner_factor_type_is_inplace});
    auto accumulate_inplace_fg = BuildAccumulateInplaceOutputFG("accumulate_dout_inplace");
    auto select_fg = BuildSelectOutputFG("accumulate_dout_select");
    auto inner_switch = inner_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimSwitch), inner_cond, NewValueNode(accumulate_inplace_fg), NewValueNode(select_fg)});
    auto inner_ret = inner_fg->NewCNodeInOrder({inner_switch, inner_dout_tuple_input, inner_factor_tuple_input});
    inner_fg->set_output(inner_ret);
    auto switch_node = fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimSwitch), add_out_cond, NewValueNode(add_output_fg), NewValueNode(inner_fg)});
    auto ret_node = fg->NewCNodeInOrder({switch_node, dout_tuple_input, factor_tuple_input});
    fg->set_output(ret_node);
    return fg;
  }

  // Constant scene, no switch.
  if (types_["dout"] == OpsType::Type_Variable || types_["factor"] == OpsType::Type_Variable) {
    return BuildChooseOutputFG("accumulate_choose_output");
  }
  if (IsAddDout()) {
    return BuildAddOutputFG("accumulate_dout_add_output", args_abs_list);
  }
  if (types_["dout"] == OpsType::Type_Inplace && types_["factor"] == OpsType::Type_Inplace) {
    return BuildAccumulateInplaceOutputFG("accumulate_inplace_dout");
  }
  return BuildSelectOutputFG("accumulate_dout_select");
}

void AccumulateDout::CheckAccumulateDoutInputAbstract(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_size = 2;
  if (args_abs_list.size() != input_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << name_ << " input size should be " << input_size << " but got "
                               << args_abs_list.size();
  }

  auto check_abstract = [this](const AbstractBasePtr &abs, const std::string &input_name) {
    MS_EXCEPTION_IF_NULL(abs);
    if (!IsDoutTupleContainsMask(abs)) {
      types_[input_name] = 0;
      return;
    }
    if (!abs->isa<abstract::AbstractTuple>()) {
      MS_LOG(INTERNAL_EXCEPTION) << input_name << " should be tuple but got " << abs->ToString();
    }
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    constexpr size_t tuple_size = 2;
    if (abs_tuple->size() != tuple_size) {
      MS_LOG(INTERNAL_EXCEPTION) << input_name << " should have " << tuple_size << " elements but got "
                                 << abs_tuple->size() << " elements.";
    }
    const auto &abs_tuple_elements = abs_tuple->elements();
    constexpr size_t inner_tuple_index = 1;
    auto inner_abs = abs_tuple_elements[inner_tuple_index];
    MS_EXCEPTION_IF_NULL(inner_abs);
    if (!inner_abs->isa<abstract::AbstractTuple>()) {
      MS_LOG(INTERNAL_EXCEPTION) << input_name << " index " << inner_tuple_index << " input should be tuple but got "
                                 << inner_abs->ToString();
    }
    auto inner_tuple_abs = inner_abs->cast<abstract::AbstractTuplePtr>();
    constexpr size_t inner_tuple_size = 2;
    if (inner_tuple_abs->size() != inner_tuple_size) {
      MS_LOG(INTERNAL_EXCEPTION) << input_name << " index " << inner_tuple_index << " tuple but got " << tuple_size
                                 << "elements but got " << inner_tuple_abs->size() << "elements.";
    }
    const auto &inner_tuple_elements = inner_tuple_abs->elements();
    constexpr size_t factor_type_index = 1;
    const auto &factor_abs = inner_tuple_elements[factor_type_index];
    MS_EXCEPTION_IF_NULL(factor_abs);
    if (factor_abs->BuildType()->type_id() != kNumberTypeInt64) {
      MS_LOG(INTERNAL_EXCEPTION) << "Factor should be int64 scalar but got " << factor_abs->ToString();
    }
    auto factor_value = factor_abs->BuildValue();
    if (factor_value != kValueAny) {
      auto type = GetValue<int64_t>(factor_value);
      if (type != OpsType::Type_Normal && type != OpsType::Type_Inplace && type != OpsType::Type_Variable) {
        MS_LOG(INTERNAL_EXCEPTION) << input_name << " type should be " << OpsType::Type_Normal << " or "
                                   << OpsType::Type_Inplace << " or " << OpsType::Type_Variable << " but got " << type;
      }
      types_[input_name] = type;
      return;
    }
    types_[input_name] = OpsType::Type_Any;
  };

  constexpr size_t dout_index = 0;
  const auto &dout_abs = args_abs_list[dout_index];
  check_abstract(dout_abs, "dout");

  constexpr size_t factor_index = 1;
  const auto &factor_abs = args_abs_list[factor_index];
  check_abstract(factor_abs, "factor");
}

bool AccumulateDout::IsAddDout() { return types_["factor"] == OpsType::Type_Normal; }

bool AccumulateDout::IsBuildSwitchNode() {
  if (types_["dout"] != OpsType::Type_Any && types_["factor"] != OpsType::Type_Any) {
    return false;
  }
  return !IsAddDout();
}

FuncGraphPtr GenerateMask::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_len = 1;
  if (args_abs_list.size() != input_len) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << name_ << ", the input length should be " << input_len << " but got "
                               << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  auto input = fg->add_parameter();
  auto input_abstract = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abstract);
  if (!input_abstract->isa<abstract::AbstractTensor>()) {
    fg->set_output(input);
    return fg;
  }
  auto type_node = NewValueNode(MakeValue<int64_t>(kBool->type_id()));
  auto ret = fg->NewCNodeInOrder({NewValueNode(prim::kPrimOnesLikeExt), input, type_node});
  fg->set_output(ret);
  return fg;
}

FuncGraphPtr GenerateBpropOutTuple::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t input_len = 1;
  if (args_abs_list.size() != input_len) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << name_ << ", the input length should be " << input_len << " but got "
                               << args_abs_list.size();
  }
  auto input_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(input_abs);
  auto fg = std::make_shared<FuncGraph>();
  auto input = fg->add_parameter();
  if (!input_abs->isa<abstract::AbstractTuple>()) {
    auto generate_bprop_mask = std::make_shared<prim::GenerateMask>("generate_bprop_mask");
    auto dout_mask = fg->NewCNodeInOrder({NewValueNode(generate_bprop_mask), input});
    auto ops_type = NewValueNode(ops_type_);
    auto bprop_inner_mask = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), dout_mask, ops_type});
    auto bprop_with_mask = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), input, bprop_inner_mask});
    fg->set_output(bprop_with_mask);
    return fg;
  }
  auto input_tuple_abs = input_abs->cast<abstract::AbstractTuplePtr>();
  AnfNodePtrList ret_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (int64_t i = 0; i < SizeToLong(input_tuple_abs->size()); ++i) {
    auto bprop_output_i = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), input, NewValueNode(i)});
    auto generate_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("generate_dout_tuple");
    generate_dout_tuple->set_ops_type(ops_type_);
    auto bprop_with_mask = fg->NewCNodeInOrder({NewValueNode(generate_dout_tuple), bprop_output_i});
    ret_inputs.push_back(bprop_with_mask);
  }
  auto ret = fg->NewCNodeInOrder(ret_inputs);
  fg->set_output(ret);
  return fg;
}

AnfNodePtr GenerateRealBpropOutput(const FuncGraphPtr &fg, const AnfNodePtr &node, const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
  if (tuple_abs == nullptr || tuple_abs->dynamic_len()) {
    return node;
  }
  // {env_type, (bprop_output, (dout_mask, ops_type))} -> {env_type, bprop_output}
  if (IsDoutTupleContainsMask(tuple_abs)) {
    constexpr int64_t index_real_bprop = 0;
    return fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(index_real_bprop)});
  }
  // It could be {env_type, (bprop_output1, (dout_mask1, ops_type1)), (bprop_output2, (dout_mask2, ops_type2)), ...}
  bool changed = false;
  AnfNodePtrList node_inputs_list{NewValueNode(prim::kPrimMakeTuple)};
  const auto &elems = tuple_abs->elements();
  for (size_t i = 0; i < elems.size(); ++i) {
    auto item_abs = elems[i];
    auto item_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(SizeToLong(i))});
    auto out_node = GenerateRealBpropOutput(fg, item_node, item_abs);
    (void)node_inputs_list.emplace_back(out_node);
    if (out_node != item_node) {
      changed = true;
    }
  }
  if (!changed) {
    return node;
  }
  return fg->NewCNodeInOrder(node_inputs_list);
}

FuncGraphPtr GetRealBpropOut::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t required_size = 1;
  if (args_abs_list.size() != required_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << name_ << ", the input length should be " << required_size << " but got "
                               << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  auto input = fg->add_parameter();
  constexpr size_t index_input = 0;
  auto real_bprop_output = GenerateRealBpropOutput(fg, input, args_abs_list[index_input]);
  fg->set_output(real_bprop_output);
  return fg;
}

FuncGraphPtr GetDependDoutTuple::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  constexpr size_t required_size = 2;
  if (args_abs_list.size() != required_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << name_ << ", the input length should be " << required_size << " but got "
                               << args_abs_list.size();
  }
  auto fg = std::make_shared<FuncGraph>();
  auto din_abs = args_abs_list[0];
  auto fg_dout_abs = args_abs_list[1];
  auto din_tuple = fg->add_parameter();
  auto fg_dout_tuple = fg->add_parameter();
  if (IsDoutTupleContainsMask(din_abs) && IsDoutTupleContainsMask(fg_dout_abs)) {
    auto fg_mask_tuple =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), fg_dout_tuple, NewValueNode(int64_t(1))});
    auto din = fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), din_tuple, NewValueNode(int64_t(0))});
    auto depend_dout_tuple = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), din, fg_mask_tuple});
    fg->set_output(depend_dout_tuple);
    return fg;
  }
  fg->set_output(fg_dout_tuple);
  return fg;
}
}  // namespace prim
}  // namespace mindspore
