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
#include "include/common/pybind_api/api_register.h"
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
#include "frontend/operator/composite/auto_generate/functional_map.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
void RegFunctional(const py::module *m) {
  (void)py::class_<Functional, std::shared_ptr<Functional>>(*m, "Functional_")
    .def(py::init<py::str &>())
    .def_property_readonly("name", &Functional::name, "Get functional name.");
}

namespace prim {
namespace {
size_t GetHashIdForFunctionalCache(const std::string &functional_name, const AbstractBasePtrList &args_abs_list,
                                   bool is_method) {
  return hash_combine(
    {std::hash<std::string>()(functional_name), AbstractBasePtrListHash(args_abs_list), std::hash<bool>{}(is_method)});
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

std::set<std::string> *GetMethodKwonlyArgs(const std::string &prim_name) {
  const auto &iter = ops::tensor_method_kwonlyargs_map.find(prim_name);
  if (iter != ops::tensor_method_kwonlyargs_map.end()) {
    return &iter->second;
  }
  return nullptr;
}

std::set<std::string> *GetFunctionKwonlyArgs(const std::string &prim_name) {
  const auto &iter = ops::function_kwonlyargs_map.find(prim_name);
  if (iter != ops::function_kwonlyargs_map.end()) {
    return &iter->second;
  }
  return nullptr;
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

std::pair<size_t, bool> GetVarargsIndex(const std::string &prim_name, bool is_method) {
  if (is_method) {
    const auto &iter = ops::tensor_method_varargs_map.find(prim_name);
    if (iter != ops::tensor_method_varargs_map.end()) {
      return std::pair<size_t, bool>(iter->second, true);
    }
    return std::pair<size_t, bool>(0, false);
  } else {
    const auto &iter = ops::function_varargs_map.find(prim_name);
    if (iter != ops::function_varargs_map.end()) {
      return std::pair<size_t, bool>(iter->second, true);
    }
    return std::pair<size_t, bool>(0, false);
  }
}

bool CheckKwargs(const std::string &prim_name, const std::map<std::string, ops::OP_DTYPE> &keyword_args_dtype,
                 const std::vector<ops::OP_DTYPE> &position_args_dtype, bool has_varargs) {
  const auto &op_def = ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  for (const auto &[key, value] : keyword_args_dtype) {
    auto op_indexes = op_def->indexes_;
    const auto &op_args = op_def->args_;
    const auto &iter = op_indexes.find(key);
    if (iter == op_indexes.end()) {
      MS_LOG(DEBUG) << "Mismatch: For Primitive[" << prim_name << "], no arg matching '" << key << "' could be found.";
      return false;
    }
    // Check key index.
    auto index_key = iter->second;
    if (index_key < position_args_dtype.size() && !has_varargs) {
      MS_LOG(DEBUG) << "Mismatch: Primitive[" << prim_name << "] got multiple values for argument '" << key << "'.";
      return false;
    }
    // Check value dtype.
    auto op_arg = op_args[index_key];
    if (!MatchPrimitiveArgDtype(prim_name, op_arg, value)) {
      return false;
    }
  }
  return true;
}

size_t GetPrimDefaultSize(const std::vector<ops::OpInputArg> &op_args, const std::string &prim_name,
                          size_t varargs_index, bool has_varargs) {
  auto default_dict = parse::GetPrimDefaultDict(prim_name);
  bool has_default = !py::isinstance<py::none>(default_dict);
  // The default value of vararg is ().
  bool vararg_non_default =
    has_varargs && ((has_default && !default_dict.contains(op_args[varargs_index].arg_name_)) || !has_default);
  size_t varargs_count = vararg_non_default ? 1 : 0;
  if (!has_default) {
    return varargs_count;
  }
  return varargs_count + default_dict.cast<py::dict>().size();
}

bool CheckPositionArgs(const std::string &prim_name, const std::vector<ops::OP_DTYPE> &position_args_dtype,
                       bool is_method, bool *need_pack) {
  size_t check_position_size = position_args_dtype.size();
  auto has_varargs_index_pair = GetVarargsIndex(prim_name, is_method);
  size_t varargs_index = has_varargs_index_pair.first;
  bool has_varargs = has_varargs_index_pair.second;
  if (has_varargs) {
    bool all_int = false;
    if (position_args_dtype.size() > varargs_index) {
      check_position_size = varargs_index;
      all_int = std::all_of(position_args_dtype.begin() + varargs_index,
                            position_args_dtype.begin() + position_args_dtype.size(),
                            [](const auto &op_dtype) { return op_dtype == ops::DT_INT; });
    }
    // all of args type show be int or primitive name has "Deprecated"
    if ((prim_name.find("Deprecated") != std::string::npos) || all_int) {
      *need_pack = true;
    }
  }
  const auto kwonly_list = is_method ? GetMethodKwonlyArgs(prim_name) : GetFunctionKwonlyArgs(prim_name);
  const auto &op_def = ops::GetOpDef(prim_name);
  const auto &op_args = op_def->args_;
  for (size_t i = 0; i < check_position_size; ++i) {
    // position argument should not be keyword-only.
    const auto &arg_name = op_args[i].arg_name_;
    if (kwonly_list != nullptr && kwonly_list->find(arg_name) != kwonly_list->end()) {
      return false;
    }
    if (!MatchPrimitiveArgDtype(prim_name, op_args[i], position_args_dtype[i])) {
      return false;
    }
  }
  return true;
}

bool MatchPrimitiveArgs(const std::string &functional_name, const std::string &prim_name,
                        const abstract::AbstractBasePtrList &args_abs_list, bool is_method, bool *need_pack) {
  const auto &op_def = ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot find OpDef of Primitive[" << prim_name << "].";
  }
  // Separate position arguments and keyword arguments.
  std::vector<ops::OP_DTYPE> position_args_dtype;
  std::map<std::string, ops::OP_DTYPE> keyword_args_dtype;
  GetOpDtypeList(prim_name, args_abs_list, &position_args_dtype, &keyword_args_dtype);
  // Check args size.
  const auto &op_args = op_def->args_;
  MS_LOG(DEBUG) << "Matching Primitive" << prim_name << "], expect args number: " << op_args.size()
                << ". The number of position args is " << position_args_dtype.size() << " and that of keyword args is "
                << keyword_args_dtype.size() << ".";
  // If no varargs , check args size
  auto inputs_size = position_args_dtype.size() + keyword_args_dtype.size();
  auto has_varargs_index_pair = GetVarargsIndex(prim_name, is_method);
  size_t varargs_index = has_varargs_index_pair.first;
  bool has_varargs = has_varargs_index_pair.second;
  if (!has_varargs && inputs_size > op_args.size()) {
    return false;
  }
  if (!CheckKwargs(prim_name, keyword_args_dtype, position_args_dtype, has_varargs)) {
    return false;
  }
  if (!CheckPositionArgs(prim_name, position_args_dtype, is_method, need_pack)) {
    return false;
  }
  // Check the number of arguments.
  auto least_size = op_args.size() - GetPrimDefaultSize(op_args, prim_name, varargs_index, has_varargs);
  if (inputs_size < least_size) {
    return false;
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

std::stringstream BuildApiInputInfo(const std::string &function_name, const std::vector<std::string> &arg_info_list) {
  std::stringstream ss;
  std::string result = std::accumulate(
    arg_info_list.begin(), arg_info_list.end(), std::string(),
    [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
  ss << "Failed calling " << function_name << " with \"" << function_name << "(" << result << ")\".\n";
  ss << "The valid calling should be:\n";
  return ss;
}

std::string BuildFunctionalErrorMsg(const std::string &function_name, const std::vector<std::string> &arg_info_list,
                                    bool is_method) {
  std::stringstream ss = BuildApiInputInfo(function_name, arg_info_list);
  const auto &signature_map =
    is_method ? ops::tensor_method_overload_signature_map : ops::function_overload_signature_map;
  auto it = signature_map.find(function_name);
  if (it != signature_map.end()) {
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
  return NormalizeTypeId(type_id) == kObjectTypeTensorType &&
         ops::tensor_method_overload_map.find(method_name) != ops::tensor_method_overload_map.end();
}

ValuePtr TransformFunctionalToPrimitive(const std::string &functional_name,
                                        const abstract::AbstractBasePtrList &args_abs_list, bool is_method,
                                        bool *need_pack) {
  // Check cache.
  auto hash_id = GetHashIdForFunctionalCache(functional_name, args_abs_list, is_method);
  const auto &cache_iter = GetFunctionalConvertCache().find(hash_id);
  if (cache_iter != GetFunctionalConvertCache().end()) {
    const auto &prim = cache_iter->second.first;
    *need_pack = cache_iter->second.second;
    MS_LOG(DEBUG) << "Get functional cache: " << functional_name << ", primitive name: " << prim->ToString();
    return prim;
  }
  // Convert Function to Primitive.
  const auto &overload_map = is_method ? ops::tensor_method_overload_map : ops::function_overload_map;
  const auto &iter = overload_map.find(functional_name);
  if (iter == overload_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Functional[" << functional_name << "] does not support overloading.";
  }
  const auto &prim_list = iter->second;
  // Find matching Primitive.
  MS_LOG(DEBUG) << "Start looking for matched primitive for Functional[" << functional_name << "].";
  ValuePtr match_prim = nullptr;
  for (const auto &prim : prim_list) {
    const auto &prim_name = GetPrimName(prim);
    if (MatchPrimitiveArgs(functional_name, prim_name, args_abs_list, is_method, need_pack)) {
      match_prim = prim;
      break;
    }
  }
  if (match_prim == nullptr) {
    std::vector<std::string> arg_info_list;
    (void)std::transform(args_abs_list.cbegin(), args_abs_list.cend(), std::back_inserter(arg_info_list),
                         [](const AbstractBasePtr &op_abs) { return BuildArgsTypeString(op_abs->BuildType()); });
    MS_EXCEPTION(TypeError) << BuildFunctionalErrorMsg(functional_name, arg_info_list, is_method);
  }
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to Primitive: " << match_prim->ToString();
  GetFunctionalConvertCache()[hash_id] = std::make_pair(match_prim, *need_pack);
  return match_prim;
}

AnfNodePtrList GeneratePrimitivePackPositionArgs(const FuncGraphPtr &func_graph,
                                                 const std::vector<AnfNodePtr> &args_list, size_t position_args_size,
                                                 size_t var_args_index) {
  AnfNodePtrList nodes;
  if (position_args_size <= var_args_index) {
    for (size_t i = 0; i < position_args_size; ++i) {
      (void)nodes.emplace_back(args_list[i]);
    }
  } else {
    for (size_t i = 0; i < var_args_index; ++i) {
      (void)nodes.emplace_back(args_list[i]);
    }
    AnfNodePtrList pack_args{NewValueNode(prim::kPrimMakeTuple)};
    (void)std::copy(args_list.begin() + var_args_index, args_list.begin() + position_args_size,
                    std::back_inserter(pack_args));
    auto pack_args_node = func_graph->NewCNodeInOrder(pack_args);
    nodes.emplace_back(pack_args_node);
  }
  return nodes;
}

void GeneratePrimitivePackKeywordArgs(const std::string &prim_name, const std::vector<ops::OpInputArg> &op_args,
                                      std::map<std::string, AnfNodePtr> *key_map, size_t var_args_index,
                                      AnfNodePtrList *nodes) {
  size_t nodes_size = nodes->size();
  for (size_t i = nodes_size; i < op_args.size(); ++i) {
    const auto &arg_name = op_args[i].arg_name_;
    const auto &iter = key_map->find(arg_name);
    if (iter != key_map->end()) {
      MS_LOG(DEBUG) << "Get args for Primitive[" << prim_name << "]: " << iter->second->DebugString();
      (void)nodes->emplace_back(iter->second);
      (void)key_map->erase(arg_name);
    } else {
      if (i == var_args_index) {
        auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
        (void)nodes->emplace_back(NewValueNode(empty_tuple_value));
        continue;
      }
      auto default_arg = parse::GetArgDefaultValue(prim_name, arg_name);
      if (default_arg == nullptr) {
        break;
      }
      MS_LOG(DEBUG) << "Get the default value of '" << arg_name << "' attribute of Primitive[" << prim_name
                    << "], which is " << default_arg->ToString() << ".";
      (void)nodes->emplace_back(NewValueNode(default_arg));
    }
  }
}

AnfNodePtrList GeneratePrimitivePackArgs(const std::pair<std::string, bool> &params,
                                         const std::vector<AnfNodePtr> &args_list,
                                         const std::vector<ops::OpInputArg> &op_args,
                                         const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                         const FuncGraphPtr &graph) {
  const std::string &prim_name = params.first;
  bool is_method = params.second;
  size_t var_args_index = GetVarargsIndex(prim_name, is_method).first;
  size_t args_size = args_list.size();
  std::map<std::string, AnfNodePtr> key_map;
  for (size_t idx = 0; idx < args_list.size(); ++idx) {
    auto input = args_list[idx];
    if (abstract::IsMonad(input)) {
      --args_size;
      continue;
    }
    auto input_abs = eval_func(input);
    if (input_abs->isa<abstract::AbstractKeywordArg>()) {
      abstract::GetKeywordArgsMap(input_abs, op_args, input, graph, &key_map);
    }
  }
  args_size -= key_map.size();
  AnfNodePtrList nodes = GeneratePrimitivePackPositionArgs(graph, args_list, args_size, var_args_index);
  GeneratePrimitivePackKeywordArgs(prim_name, op_args, &key_map, var_args_index, &nodes);

  if (nodes.size() != op_args.size()) {
    std::string args_type_str = (op_args.size() != 0 && op_args[0].as_init_arg_) ? "init arguments" : "inputs";
    MS_EXCEPTION(TypeError) << "For Operator[" << prim_name << "], the number of " << args_type_str
                            << " (including default arguments) should be " << op_args.size()
                            << ", but the actual number of inputs is not satisfied, which is " << args_size << ".";
  }
  return nodes;
}

AnfNodePtr ConvertFunctionalToPrimitive(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                        bool is_method) {
  bool need_pack = false;
  auto prim = TransformFunctionalToPrimitive(functional_name, args_abs_list, is_method, &need_pack);
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
    args_node_list = GeneratePrimitivePackArgs(std::make_pair(prim_name, is_method), inputs_list, op_def->args_,
                                               eval_func, func_graph);
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
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        bool is_method) {
  MS_LOG(DEBUG) << "Functional[" << functional_name << "] receives arguments that contain Any.";
  std::stringstream script_buffer;
  constexpr auto index_fn = 0;
  std::string data_str = fallback::ConvertRealStrToUnicodeStr(functional_name, index_fn);
  auto fn_node = inputs_list[index_fn];
  auto fn_key_node = NewValueNode(std::make_shared<StringImm>(data_str));
  std::vector<AnfNodePtr> keys_inputs{NewValueNode(prim::kPrimMakeTuple), fn_key_node};
  std::vector<AnfNodePtr> values_inputs{NewValueNode(prim::kPrimMakeTuple), fn_node};
  if (is_method) {
    // Convert Functional node to PyExectue("x.method_name(arg1, arg2)", local_keys, local_values)
    script_buffer << data_str << "." << functional_name + "(";
  } else {
    // Convert Functional node to PyExectue("mint.xxx(x, arg1, arg2)", local_keys, local_values)
    script_buffer << functional_name << "(" << data_str << ", ";
    py::object py_functional = python_adapter::GetPyFn("mindspore.mint", functional_name);
    if (py::isinstance<py::none>(py_functional)) {
      MS_LOG(INTERNAL_EXCEPTION) << functional_name << " is not a function of mindspore.mint";
    }
    auto functional_node = NewValueNode(std::make_shared<parse::InterpretedObject>(py_functional));
    (void)keys_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>(functional_name)));
    (void)values_inputs.emplace_back(functional_node);
  }
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
