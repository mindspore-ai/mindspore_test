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

#include "frontend/operator/composite/functional_overload.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/ops/op_def/structure_ops.h"
#include "ops/op_def.h"
#include "ir/core_ops_primitive.h"
#include "frontend/operator/ops.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/primfunc_utils.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "pipeline/pynative/op_function/auto_generate/functional_map.h"

namespace mindspore {
void RegFunctional(const py::module *m) {
  (void)py::class_<Functional, std::shared_ptr<Functional>>(*m, "Functional_")
    .def(py::init<py::str &, py::bool_ &>())
    .def_property_readonly("name", &Functional::name, "Get functional name.");
}

namespace prim {
namespace {
size_t GetHashIdForFunctionalCache(const std::string &functional_name, const AbstractBasePtrList &args_abs_list) {
  return hash_combine(std::hash<std::string>()(functional_name), AbstractBasePtrListHash(args_abs_list));
}

bool MatchExpectedDtype(const ops::OP_DTYPE &input_dtype, const ops::OP_DTYPE &expected_dtype) {
  MS_LOG(DEBUG) << "Input dtype is: '" << ops::EnumToString(input_dtype) << "' and expected dtype is '"
                << ops::EnumToString(expected_dtype) << "'.";
  // Check if the types match.
  if (input_dtype == expected_dtype || expected_dtype == ops::OP_DTYPE::DT_ANY) {
    return true;
  }
  static std::set<ops::OP_DTYPE> number_dtype_list = {ops::OP_DTYPE::DT_BOOL, ops::OP_DTYPE::DT_INT,
                                                      ops::OP_DTYPE::DT_FLOAT};
  static std::set<ops::OP_DTYPE> tuple_dtype_list = {ops::OP_DTYPE::DT_TUPLE_BOOL,   ops::OP_DTYPE::DT_TUPLE_INT,
                                                     ops::OP_DTYPE::DT_TUPLE_FLOAT,  ops::OP_DTYPE::DT_TUPLE_NUMBER,
                                                     ops::OP_DTYPE::DT_TUPLE_TENSOR, ops::OP_DTYPE::DT_TUPLE_STR,
                                                     ops::OP_DTYPE::DT_TUPLE_ANY};
  static std::set<ops::OP_DTYPE> list_dtype_list = {ops::OP_DTYPE::DT_LIST_BOOL,   ops::OP_DTYPE::DT_LIST_INT,
                                                    ops::OP_DTYPE::DT_LIST_FLOAT,  ops::OP_DTYPE::DT_LIST_NUMBER,
                                                    ops::OP_DTYPE::DT_LIST_TENSOR, ops::OP_DTYPE::DT_LIST_STR,
                                                    ops::OP_DTYPE::DT_LIST_ANY};
  // Check number.
  if (expected_dtype == ops::OP_DTYPE::DT_NUMBER && number_dtype_list.find(input_dtype) != number_dtype_list.end()) {
    return true;
  }
  // Check Tuple without checking its elements.
  if (input_dtype == ops::OP_DTYPE::DT_TUPLE_ANY && tuple_dtype_list.find(expected_dtype) != tuple_dtype_list.end()) {
    return true;
  }
  // Check List without checking its elements.
  if (input_dtype == ops::OP_DTYPE::DT_LIST_ANY && list_dtype_list.find(expected_dtype) != tuple_dtype_list.end()) {
    return true;
  }
  return false;
}

bool MatchPrimitiveArgDtype(const std::string &prim_name, const ops::OpInputArg &op_arg,
                            const ops::OP_DTYPE &input_dtype) {
  MS_LOG(DEBUG) << "Matching arg '" << op_arg.arg_name_ << "' for Primitive[" << prim_name << "] with dtype "
                << ops::EnumToString(input_dtype) << ".";
  if (MatchExpectedDtype(input_dtype, op_arg.arg_dtype_) ||
      (op_arg.is_optional_ && input_dtype == ops::OP_DTYPE::DT_NONE)) {
    return true;
  }
  if (!op_arg.cast_dtype_.empty()) {
    return std::any_of(op_arg.cast_dtype_.cbegin(), op_arg.cast_dtype_.cend(),
                       [&input_dtype](const ops::OP_DTYPE &dtype) { return MatchExpectedDtype(input_dtype, dtype); });
  }
  if (!op_arg.arg_handler_.empty()) {
    auto src_dtypes = ops::GetSourceDtypeByArgHandler(op_arg.arg_handler_);
    return std::any_of(src_dtypes.cbegin(), src_dtypes.cend(),
                       [&input_dtype](const ops::OP_DTYPE &dtype) { return MatchExpectedDtype(input_dtype, dtype); });
  }
  return false;
}

ops::OP_DTYPE GetOpDtypeFromAbstract(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  const auto &abs_type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  if (abs->isa<abstract::AbstractTensor>()) {
    return ops::OP_DTYPE::DT_TENSOR;
  }
  if (abs->isa<abstract::AbstractTuple>()) {
    return ops::OP_DTYPE::DT_TUPLE_ANY;
  }
  if (abs->isa<abstract::AbstractList>()) {
    return ops::OP_DTYPE::DT_LIST_ANY;
  }
  if (abs->isa<abstract::AbstractNone>()) {
    return ops::OP_DTYPE::DT_NONE;
  }
  if (abs->isa<abstract::AbstractType>() && abs_type->isa<Type>()) {
    return ops::OP_DTYPE::DT_TYPE;
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    if (abs_type->isa<Bool>()) {
      return ops::OP_DTYPE::DT_BOOL;
    }
    if (abs_type->isa<Int>() || abs_type->isa<UInt>()) {
      return ops::OP_DTYPE::DT_INT;
    }
    if (abs_type->isa<Float>() || abs_type->isa<BFloat>()) {
      return ops::OP_DTYPE::DT_FLOAT;
    }
    if (abs_type->isa<Number>()) {
      return ops::OP_DTYPE::DT_NUMBER;
    }
    if (abs_type->isa<String>()) {
      return ops::OP_DTYPE::DT_STR;
    }
  }
  return ops::OP_DTYPE::DT_ANY;
}

void GetOpDtypeList(const std::string &prim_name, const abstract::AbstractBasePtrList &args_abs_list,
                    std::vector<ops::OP_DTYPE> *position_args_dtype,
                    std::map<std::string, ops::OP_DTYPE> *keyword_args_dtype) {
  for (const auto &abs : args_abs_list) {
    // Ignore monad.
    if (abs->isa<abstract::AbstractMonad>()) {
      continue;
    }
    if (abs->isa<abstract::AbstractKeywordArg>()) {
      auto kw_abs = abs->cast<abstract::AbstractKeywordArgPtr>();
      const std::string &key = kw_abs->get_key();
      if (keyword_args_dtype->find(key) != keyword_args_dtype->end()) {
        MS_EXCEPTION(TypeError) << "Primitive[" << prim_name << "] got multiple values for argument '" << key << "'";
      }
      auto op_dtype = GetOpDtypeFromAbstract(kw_abs->get_arg());
      keyword_args_dtype->insert(std::make_pair(key, op_dtype));
    } else {
      auto op_dtype = GetOpDtypeFromAbstract(abs);
      (void)position_args_dtype->emplace_back(op_dtype);
    }
  }
}

bool MatchPrimitiveWithPackArgs(const ops::OpDefPtr &op_def, const std::vector<ops::OP_DTYPE> &position_args_dtype,
                                bool *need_pack) {
  // For example: Reshape(tensor, 2, 3) -> Reshape(tensor, (2, 3))
  constexpr auto requeired_size = 2;
  constexpr auto index_allow_pack = 1;
  const auto &op_args = op_def->args_;
  bool allow_pack = op_args.size() == requeired_size && position_args_dtype.size() >= requeired_size &&
                    (op_args[index_allow_pack].arg_dtype_ == ops::DT_TUPLE_INT ||
                     op_args[index_allow_pack].arg_dtype_ == ops::DT_LIST_INT);
  if (!allow_pack) {
    return false;
  }
  bool all_int = std::all_of(position_args_dtype.begin() + index_allow_pack, position_args_dtype.end(),
                             [](const auto &op_dtype) { return op_dtype == ops::DT_INT; });
  if (all_int) {
    *need_pack = true;
  }
  return all_int;
}

bool MatchPrimitiveArgs(const std::string &prim_name, const abstract::AbstractBasePtrList &args_abs_list,
                        bool *need_pack) {
  const auto &op_def = ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot find OpDef of Primitive[" << prim_name << "].";
  }
  // Separate position arguments and keyword arguments.
  std::vector<ops::OP_DTYPE> position_args_dtype;
  std::map<std::string, ops::OP_DTYPE> keyword_args_dtype;
  GetOpDtypeList(prim_name, args_abs_list, &position_args_dtype, &keyword_args_dtype);
  // If there is a single positional IntArray argument, allow a var-args stype IntArray,
  // so x.reshape(2, 3) behaves as x.reshape((2, 3)).
  if (keyword_args_dtype.empty() && MatchPrimitiveWithPackArgs(op_def, position_args_dtype, need_pack)) {
    return true;
  }
  // Check args size.
  const auto &op_args = op_def->args_;
  MS_LOG(DEBUG) << "Matching Primitive" << prim_name << "], expect args number: " << op_args.size()
                << ". The number of position args is " << position_args_dtype.size() << " and that of keyword args is "
                << keyword_args_dtype.size() << ".";
  if (position_args_dtype.size() + keyword_args_dtype.size() > op_args.size()) {
    return false;
  }
  // Check keyword arguments.
  for (const auto &[key, value] : keyword_args_dtype) {
    auto op_indexes = op_def->indexes_;
    const auto &iter = op_indexes.find(key);
    if (iter == op_indexes.end()) {
      MS_LOG(DEBUG) << "Mismatch: For Primitive[" << prim_name << "], no arg matching '" << key << "' could be found.";
      return false;
    }
    // Check key index.
    auto index_key = iter->second;
    if (index_key < position_args_dtype.size()) {
      MS_LOG(DEBUG) << "Mismatch: Primitive[" << prim_name << "] got multiple values for argument '" << key << "'.";
      return false;
    }
    // Check value dtype.
    auto op_arg = op_args[index_key];
    if (!MatchPrimitiveArgDtype(prim_name, op_arg, value)) {
      return false;
    }
  }
  // Check position arguments.
  for (size_t i = 0; i < position_args_dtype.size(); ++i) {
    if (!MatchPrimitiveArgDtype(prim_name, op_args[i], position_args_dtype[i])) {
      return false;
    }
  }
  return true;
}

std::string GetPrimName(const ValuePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<Primitive>()) {
    return prim->cast<PrimitivePtr>()->name();
  }
  if (prim->isa<DeprecatedTensorMethod>()) {
    return prim->cast<DeprecatedTensorMethodPtr>()->name();
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Expect Primitive or MetaFuncGraph, but got " << prim->ToString();
}

std::string BuildOtherTypeString(const TypePtr &arg_type) {
  std::stringstream ss;
  if (arg_type->isa<Keyword>()) {
    auto kw_type = arg_type->cast_ptr<Keyword>();
    ss << kw_type->GetKey() << "=" << BuildArgsTypeString(kw_type->GetValue());
    return ss.str();
  }
  if (arg_type->isa<Tuple>()) {
    auto tuple_type = arg_type->cast_ptr<Tuple>();
    if (tuple_type->dynamic_len()) {
      return "tuple";
    }
    ss << "tuple<";
    for (size_t i = 0; i < tuple_type->size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << BuildArgsTypeString(tuple_type->elements()[i]);
    }
    ss << ">";
    return ss.str();
  }
  if (arg_type->isa<List>()) {
    auto list_type = arg_type->cast_ptr<List>();
    if (list_type->dynamic_len()) {
      return "list";
    }
    ss << "list<";
    for (size_t i = 0; i < list_type->size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << BuildArgsTypeString(list_type->elements()[i]);
    }
    ss << ">";
    return ss.str();
  }
  return arg_type->ToString();
}
}  // namespace

std::string BuildArgsTypeString(const TypePtr &arg_type) {
  MS_EXCEPTION_IF_NULL(arg_type);
  if (arg_type->isa<Bool>()) {
    return "bool";
  }
  if (arg_type->isa<Int>() || arg_type->isa<UInt>()) {
    return "int";
  }
  if (arg_type->isa<Float>() || arg_type->isa<BFloat>()) {
    return "float";
  }
  if (arg_type->isa<String>()) {
    return "string";
  }
  if (arg_type->isa<TypeNone>()) {
    return "None";
  }
  if (arg_type->isa<TensorType>()) {
    return "Tensor";
  }
  return BuildOtherTypeString(arg_type);
}

std::string BuildFunctionalErrorMsg(const std::string &function_name, const std::vector<std::string> &arg_info_list) {
  std::string result = std::accumulate(
    arg_info_list.begin(), arg_info_list.end(), std::string(),
    [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
  std::stringstream ss;
  ss << "Failed calling " << function_name << " with \"" << function_name << "(" << result << ")\".\n";
  ss << "The valid calling should be:\n";
  auto it = ops::func_signature_map.find(function_name);
  if (it != ops::func_signature_map.end()) {
    const std::vector<std::string> &valid_arg_options = it->second;
    for (const std::string &arg_option : valid_arg_options) {
      ss << "\"" << arg_option << "\"\n";
    }
    ss << std::endl;
  } else {
    MS_LOG(EXCEPTION) << "Valid arg options are not correctly generated." << std::endl;
  }
  return ss.str();
}

FuncGraphPtr DeprecatedTensorMethod::GenerateFuncGraph(const abstract::AbstractBasePtrList &) {
  static const std::string module_path = "mindspore._extends.parse.deprecated.deprecated_tensor_method";
  static const std::string method_map = "deprecated_tensor_method_map";
  py::dict map_obj = python_adapter::GetPyFn(module_path, method_map);
  const auto &method_name = method();
  if (!map_obj.contains(py::str(method_name))) {
    MS_LOG(INTERNAL_EXCEPTION) << "As a deprecated Tensor method, '" << method_name
                               << "' should be registered in _extends/parse/deprecated/deprecated_tensor_method.py::"
                               << method_map;
  }
  const std::string &function_name = map_obj[py::str(method_name)].cast<std::string>();
  auto value = prim::GetPythonOps(function_name, parse::PYTHON_MOD_OPS_TENSOR_METHOD_MODULE);
  if (!value->isa<FuncGraph>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Expect FuncGraph, but got " << value->ToString();
  }
  return value->cast<FuncGraphPtr>();
}

std::map<size_t, std::pair<ValuePtr, bool>> &GetFunctionalConvertCache() {
  static std::map<size_t, std::pair<ValuePtr, bool>> functional_convert_cache;
  return functional_convert_cache;
}

bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name) {
  // Check if tensor.
  if (NormalizeTypeId(type_id) != kObjectTypeTensorType ||
      ops::functional_method_map.find(method_name) == ops::functional_method_map.end()) {
    return false;
  }
  // Check for duplicate definitions.
  if (!pipeline::Resource::GetMethodPtr(type_id, method_name).empty() ||
      !pipeline::Resource::GetAttrPtr(type_id, method_name).empty()) {
    MS_LOG(INTERNAL_EXCEPTION)
      << "There are duplicate definitions of Tensor." << method_name
      << " in graph mode. Please remove the definition in mindspore/ccsrc/pipeline/jit/ps/resource.cc";
  }
  return true;
}

ValuePtr TransformFunctionalToPrimitive(const std::string &functional_name,
                                        const abstract::AbstractBasePtrList &args_abs_list, bool *need_pack) {
  // Check cache.
  auto hash_id = GetHashIdForFunctionalCache(functional_name, args_abs_list);
  const auto &cache_iter = GetFunctionalConvertCache().find(hash_id);
  if (cache_iter != GetFunctionalConvertCache().end()) {
    const auto &prim = cache_iter->second.first;
    *need_pack = cache_iter->second.second;
    MS_LOG(DEBUG) << "Get functional cache: " << functional_name << ", primitive name: " << prim->ToString();
    return prim;
  }
  // Convert Function to Primitive.
  const auto &iter = ops::functional_method_map.find(functional_name);
  if (iter == ops::functional_method_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Functional[" << functional_name << "] does not support overloading.";
  }
  // Find matching Primitive.
  MS_LOG(DEBUG) << "Start looking for matched primitive for Functional[" << functional_name << "].";
  ValuePtr match_prim = nullptr;
  auto prim_list = iter->second;
  for (const auto &prim : prim_list) {
    const auto &prim_name = GetPrimName(prim);
    if (MatchPrimitiveArgs(prim_name, args_abs_list, need_pack)) {
      match_prim = prim;
      break;
    }
  }
  if (match_prim == nullptr) {
    std::vector<std::string> arg_info_list;
    (void)std::transform(args_abs_list.cbegin(), args_abs_list.cend(), std::back_inserter(arg_info_list),
                         [](const AbstractBasePtr &op_abs) { return BuildArgsTypeString(op_abs->BuildType()); });
    MS_EXCEPTION(TypeError) << BuildFunctionalErrorMsg(functional_name, arg_info_list);
  }
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to Primitive: " << match_prim->ToString();
  GetFunctionalConvertCache()[hash_id] = std::make_pair(match_prim, *need_pack);
  return match_prim;
}

AnfNodePtrList GeneratePrimitivePackArgs(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                         const ops::OpDefPtr &op_def, const FuncGraphPtr &func_graph) {
  constexpr auto index_data = 0;
  constexpr auto index_args = 1;
  AnfNodePtrList pack_args{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(inputs_list.begin() + index_args, inputs_list.end(), std::back_inserter(pack_args));
  auto pack_args_node = func_graph->NewCNodeInOrder(pack_args);
  return {inputs_list[index_data], pack_args_node};
}

AnfNodePtr ConvertFunctionalToPrimitive(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func) {
  bool need_pack = false;
  auto prim = TransformFunctionalToPrimitive(functional_name, args_abs_list, &need_pack);
  auto prim_name = GetPrimName(prim);
  const auto &op_def = ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList prim_inputs_list;
  if (prim->isa<Primitive>()) {
    auto do_trans = std::make_shared<prim::DoTransPrimitiveFunction>(prim->cast<PrimitivePtr>());
    (void)prim_inputs_list.emplace_back(NewValueNode(do_trans));
  } else {
    (void)prim_inputs_list.emplace_back(NewValueNode(prim));
  }
  AnfNodePtrList args_node_list;
  if (need_pack) {
    args_node_list = GeneratePrimitivePackArgs(functional_name, inputs_list, op_def, func_graph);
  } else {
    args_node_list =
      abstract::GeneratePrimitiveDefaultArgs(prim_name, inputs_list, op_def->args_, eval_func, func_graph);
  }
  (void)std::copy(args_node_list.begin(), args_node_list.end(), std::back_inserter(prim_inputs_list));
  auto prim_node = func_graph->NewCNodeInOrder(prim_inputs_list);
  prim_node->set_debug_info(cnode->debug_info());
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "]: " << cnode->DebugString()
                << " to Primitive: " << prim_node->DebugString();
  return prim_node;
}

AnfNodePtr ConvertFunctionalToPyExecute(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode) {
  // Convert Functional node to PyExectue("x.method_name(arg1, arg2)", local_keys, local_values)
  std::stringstream script_buffer;
  constexpr auto index_fn = 0;
  std::string data_str = fallback::ConvertRealStrToUnicodeStr(functional_name, index_fn);
  script_buffer << data_str << "." << functional_name + "(";
  auto fn_node = inputs_list[index_fn];
  auto fn_key_node = NewValueNode(std::make_shared<StringImm>(data_str));
  std::vector<AnfNodePtr> keys_inputs{NewValueNode(prim::kPrimMakeTuple), fn_key_node};
  std::vector<AnfNodePtr> values_inputs{NewValueNode(prim::kPrimMakeTuple), fn_node};
  for (size_t index = 1; index < inputs_list.size(); ++index) {
    auto internal_arg = fallback::ConvertRealStrToUnicodeStr(functional_name, index);
    if (args_abs_list[index]->isa<abstract::AbstractKeywordArg>()) {
      auto key = args_abs_list[index]->cast<abstract::AbstractKeywordArgPtr>()->get_key();
      script_buffer << key << "=" << internal_arg << ", ";
    } else {
      script_buffer << internal_arg << ", ";
    }
    (void)keys_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>(internal_arg)));
    (void)values_inputs.emplace_back(inputs_list[index]);
  }
  script_buffer << ")";
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to script: " << script_buffer.str();
  auto script_node = NewValueNode(std::make_shared<StringImm>(script_buffer.str()));
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto keys_tuple_node = fg->NewCNodeInOrder(keys_inputs);
  auto values_tuple_node = fg->NewCNodeInOrder(values_inputs);
  auto pyexecute_node =
    fallback::CreatePyExecuteCNodeInOrder(fg, script_node, keys_tuple_node, values_tuple_node, cnode->debug_info());
  pyexecute_node->set_debug_info(cnode->debug_info());
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "]: " << cnode->DebugString()
                << " to PyExecute: " << pyexecute_node->DebugString();
  return pyexecute_node;
}
}  // namespace prim
}  // namespace mindspore
