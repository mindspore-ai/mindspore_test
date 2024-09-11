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

#include "pipeline/jit/ps/static_analysis/functional_utils.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
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
    .def(py::init<py::str &>())
    .def_property_readonly("name", &Functional::name, "Get functional name.");
}

namespace prim {
namespace {
size_t GetHashIdForFunctionalCache(const std::string &functional_name, const AbstractBasePtrList &args_abs_list) {
  return hash_combine(std::hash<std::string>()(functional_name), AbstractBasePtrListHash(args_abs_list));
}

std::string BuildFunctionalErrorMsg(const std::string &functional_name, const std::vector<std::string> &op_type_list) {
  // Get inputs format.
  std::stringstream inputs_ss;
  for (const auto &type : op_type_list) {
    inputs_ss << type << ", ";
  }
  constexpr size_t truncate_offset = 2;
  auto inputs_str = inputs_ss.str();
  inputs_str = inputs_str.empty() ? "" : inputs_str.replace(inputs_str.end() - truncate_offset, inputs_str.end(), "");
  // Error message.
  std::stringstream ss;
  ss << "Failed calling " << functional_name << " with \"" << functional_name << "(" << inputs_str << ")\".\n";
  // A list of correct candiadtes should be provided later.
  return ss.str();
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
      std::string key = kw_abs->get_key();
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

bool MatchPrimitiveArgs(const std::string &prim_name, const abstract::AbstractBasePtrList &args_abs_list) {
  const auto &op_def = ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot find OpDef of Primitive[" << prim_name << "].";
  }
  // Separate position arguments and keyword arguments.
  std::vector<ops::OP_DTYPE> position_args_dtype;
  std::map<std::string, ops::OP_DTYPE> keyword_args_dtype;
  GetOpDtypeList(prim_name, args_abs_list, &position_args_dtype, &keyword_args_dtype);
  // Check args size.
  auto op_args = op_def->args_;
  MS_LOG(DEBUG) << "Matching Primitive" << prim_name << "], expect args number: " << op_args.size()
                << ". The number of position args is " << position_args_dtype.size() << " and that of keyword args is "
                << keyword_args_dtype.size() << ".";
  if (position_args_dtype.size() + keyword_args_dtype.size() > op_args.size()) {
    return false;
  }
  // Check keyword arguments.
  for (const auto &[key, value] : keyword_args_dtype) {
    auto op_indexes = op_def->indexes_;
    auto iter = op_indexes.find(key);
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
}  // namespace

std::map<size_t, std::pair<std::string, std::string>> &GetFunctionalConvertCache() {
  static std::map<size_t, std::pair<std::string, std::string>> functional_convert_cache;
  return functional_convert_cache;
}

bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name) {
  // Check if tensor.
  if (NormalizeTypeId(type_id) != kObjectTypeTensorType ||
      ops::functional_convert_map.find(method_name) == ops::functional_convert_map.end()) {
    return false;
  }
  // Check for duplicate definitions.
  if (!pipeline::Resource::GetMethodPtr(type_id, method_name).empty() ||
      !pipeline::Resource::GetAttrPtr(type_id, method_name).empty()) {
    MS_LOG(INTERNAL_EXCEPTION)
      << "There are duplicate definitions of Tensor." << method_name
      << " in graph mode. Please remove the definition in mindspore/ccsrc/pipeline/jit/ps/resource.cc.";
  }
  return true;
}

ValuePtr GetTensorPyMethod(const std::string &prim_name, const std::string &method_name) {
  if (method_name.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Primitive[" << prim_name << "] has not defined py_method.";
  }
  return prim::GetPythonOps(method_name, parse::PYTHON_MOD_OPS_TENSOR_METHOD_MODULE);
}

std::pair<std::string, std::string> ConvertFunctionalToPrimitive(const std::string &functional_name,
                                                                 const abstract::AbstractBasePtrList &args_abs_list) {
  // Check cache.
  auto hash_id = GetHashIdForFunctionalCache(functional_name, args_abs_list);
  auto cache_iter = GetFunctionalConvertCache().find(hash_id);
  if (cache_iter != GetFunctionalConvertCache().end()) {
    MS_LOG(DEBUG) << "Get functional cache: " << functional_name << ". Primitive name: " << cache_iter->second.first
                  << ", py::method: " << cache_iter->second.second;
    return cache_iter->second;
  }
  // Convert Function to Primitive.
  auto iter = ops::functional_convert_map.find(functional_name);
  if (iter == ops::functional_convert_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "No matching Functional[" << functional_name << "] found.";
  }
  auto prim_list = iter->second;
  if (prim_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Functional[" << functional_name << "] is missing primitive list.";
  }
  // Find matching Primitive.
  MS_LOG(DEBUG) << "Start looking for matched primitive for Functional[" << functional_name << "].";
  std::string match_prim_name;
  std::string match_py_method;
  for (const auto &[prim_name, py_method] : prim_list) {
    if (MatchPrimitiveArgs(prim_name, args_abs_list)) {
      match_prim_name = prim_name;
      match_py_method = py_method;
      break;
    }
  }
  if (match_prim_name.empty()) {
    std::vector<std::string> op_type_list;
    (void)std::transform(args_abs_list.cbegin(), args_abs_list.cend(), std::back_inserter(op_type_list),
                         [](const AbstractBasePtr &op_abs) { return BuildArgsTypeString(op_abs); });
    MS_EXCEPTION(TypeError) << BuildFunctionalErrorMsg(functional_name, op_type_list);
  }
  GetFunctionalConvertCache()[hash_id] = std::make_pair(match_prim_name, match_py_method);
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to Primitive[" << match_prim_name
                << "], py_method: " << match_py_method;
  return {match_prim_name, match_py_method};
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
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "]: " << cnode->DebugString()
                << " to PyExecute: " << pyexecute_node->DebugString();
  return pyexecute_node;
}
}  // namespace prim
}  // namespace mindspore
