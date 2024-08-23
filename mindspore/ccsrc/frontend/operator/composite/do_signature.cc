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

#include "frontend/operator/composite/do_signature.h"
#include <algorithm>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "include/common/utils/primfunc_utils.h"
#include "include/common/amp/amp.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ops/op_def.h"
#include "mindspore/core/utils/flags.h"
#include "mindspore/ops/op_def/arithmetic_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
namespace {
const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>() && function->cast<PrimitivePtr>()->has_signature()) {
    return function->cast<PrimitivePtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const std::string &func_name, size_t actual_param_number, const std::vector<Signature> &signature,
                    bool has_var, std::vector<AnfNodePtr> *op_inputs) {
  std::size_t sig_size = signature.size();
  auto positional_size = sig_size;
  if (has_var) {
    positional_size = sig_size - 1;
  }
  if (actual_param_number < positional_size) {
    for (size_t i = actual_param_number; i < sig_size; ++i) {
      auto default_value = signature[i].default_value;
      if (default_value == nullptr) {
        MS_LOG(EXCEPTION) << "For '" << func_name << "', the size of input should be " << sig_size << ", but got "
                          << actual_param_number << ". Please check inputs of the operator.";
      } else {
        (*op_inputs).push_back(NewValueNode(default_value));
      }
    }
  }
}

void GetTypeInfo(const std::vector<TypePtr> &input_types, std::vector<TypeId> *args_type_id,
                 std::vector<bool> *args_has_tensor) {
  for (const auto &arg_type : input_types) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (arg_type->isa<Number>()) {
      (void)args_type_id->emplace_back(arg_type->cast<NumberPtr>()->type_id());
      (void)args_has_tensor->emplace_back(false);
    } else if (arg_type->isa<TensorType>()) {
      auto elem_type = arg_type->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(elem_type);
      (void)args_type_id->emplace_back(elem_type->type_id());
      (void)args_has_tensor->emplace_back(true);
    } else {
      (void)args_type_id->emplace_back(kTypeUnknown);
      (void)args_has_tensor->emplace_back(false);
    }
  }
}

bool IsTypeRef(const SignatureEnumRW &sig, const TypePtr &type) {
  return sig == SignatureEnumRW::kRWWrite &&
         !((type->type_id() == kObjectTypeRef) || (type->type_id() == kObjectTypeRefKey));
}

void CheckSigSize(const ValuePtr &function, const size_t &sig_size, const bool &has_var,
                  const AbstractBasePtrList &args_abs_list, const std::string &func_name) {
  if (sig_size > 0) {
    if (has_var) {
      if (sig_size - 1 > args_abs_list.size()) {
        MS_LOG(EXCEPTION) << "Function " << func_name
                          << "'s input length less than PositionalKeyword Signature length.";
      }
      return;
    }
    // Consider the case where there are monads in primitive's args_abs_list.
    size_t args_size = args_abs_list.size();
    if (function->isa<Primitive>()) {
      auto prim = function->cast<PrimitivePtr>();
      if (prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) || prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO)) {
        args_size -= GetAbstractMonadNum(args_abs_list);
      }
    }
    if (args_size > sig_size) {
      MS_LOG(EXCEPTION) << "Function " << func_name << "'s input length is not equal to Signature length.";
    }
  }
}

void CheckPrimInputType(const ValuePtr &function, const AbstractBasePtrList &args_abs_list) {
  if (!function->isa<Primitive>()) {
    return;
  }
  auto prim = function->cast<PrimitivePtr>();
  const auto &prim_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    return;
  }
  auto op_args = op_def->args_;
  std::vector<ops::OpInputArg> op_call_args;
  (void)std::copy_if(op_args.cbegin(), op_args.cend(), std::back_inserter(op_call_args),
                     [](const ops::OpInputArg &arg) { return !arg.as_init_arg_; });
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    auto abs = args_abs_list[i];
    auto op_arg = op_call_args[i];
    if (!op_arg.arg_handler_.empty()) {
      continue;
    }
    if (abs->isa<abstract::AbstractKeywordArg>()) {
      continue;
    }
    if (abstract::ValidateArgSpecialType(prim_name, abs, op_arg)) {
      continue;
    }
    auto cast_dtypes = op_arg.cast_dtype_;
    bool match = std::any_of(cast_dtypes.cbegin(), cast_dtypes.cend(),
                             [&abs](const ops::OP_DTYPE &dtype) { return ops::ValidateArgsType(abs, dtype); });
    if (!match) {
      MS_EXCEPTION(TypeError) << ops::BuildOpInputsErrorMsg(op_def, op_arg.arg_name_, abs->BuildType());
    }
  }
}

SignatureEnumRW GetSignatureEnumRW(size_t index, const std::vector<Signature> &signature, bool has_var) {
  SignatureEnumRW sig = SignatureEnumRW::kRWDefault;
  // If sig_size is 0 use default.
  std::size_t sig_size = signature.size();
  if (index < sig_size) {
    sig = signature[index].rw;
  } else if (has_var && index >= sig_size) {
    sig = signature[sig_size - 1].rw;
  }
  return sig;
}

TypePtr GetMixedPrecisionTargetType(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP32)) {
    return kFloat32;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP16)) {
    return kFloat16;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_BF16)) {
    return kBFloat16;
  } else {
    return nullptr;
  }
}

bool GetImplicitPromoteType(const std::vector<Signature> &signature, const std::set<size_t> &write_indices,
                            std::vector<TypeId> *args_type_id, std::vector<bool> *args_is_tensor) {
  if (signature.empty()) {
    return false;
  }
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int64_t empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (static_cast<int64_t>(dtypes.size()) == empty_dtype_count) {
    return false;
  }
  auto args_size = dtypes.size();
  if (args_size > args_type_id->size()) {
    // It is possible that op_inputs size is larger than signatures size in vmap.
    MS_LOG(INTERNAL_EXCEPTION) << "For auto type cast, the number of args should be greater than or equal to "
                               << args_size << ", but got " << args_type_id->size() << ".";
  }
  auto sig_type_map = GetSignatureTypeMap(dtypes, *args_type_id, *args_is_tensor, write_indices);
  for (size_t i = 0; i < args_size; ++i) {
    auto it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      continue;
    }
    (*args_type_id)[i] = (it->second).first;
    (*args_is_tensor)[i] = (it->second).second;
  }
  return true;
}

bool GetAutoMixedPrecisionType(const FuncGraphPtr &func_graph, const ValuePtr &function,
                               const std::pair<std::vector<TypeId>, std::vector<bool>> &source_type_info,
                               std::vector<TypeId> *target_type_id, std::vector<bool> *target_is_tensor) {
  if (!function->isa<Primitive>() || func_graph->amp_strategy() == nullptr || !func_graph->amp_strategy()->IsEnable()) {
    return false;
  }
  const auto &prim_name = function->cast<PrimitivePtr>()->name();
  auto strategy_info = GetPrimCastStrategyInfo(func_graph->amp_strategy(), prim_name);
  if (strategy_info.strategy == amp::PrimCastStrategy::Ignore) {
    return false;
  }
  TypeId amp_type_id;
  if (strategy_info.strategy == amp::PrimCastStrategy::AutoPromote) {
    amp_type_id = GetMixPrecisionPromoteType(source_type_info.first, source_type_info.second);
    if (amp_type_id == kTypeUnknown) {
      return false;
    }
  } else {
    amp_type_id = strategy_info.dtype->type_id();
  }
  MS_LOG(DEBUG) << "For Operator[" << prim_name << "], its amp strategy is " << strategy_info.strategy
                << ", and target type_id is" << TypeIdToString(amp_type_id);

  for (size_t i = 0; i < target_type_id->size(); ++i) {
    if (IsFloatTensor((*target_type_id)[i], (*target_is_tensor)[i])) {
      (*target_type_id)[i] = amp_type_id;
    }
  }
  return true;
}

void DoTypeCast(const FuncGraphPtr &func_graph, const std::pair<ValuePtr, std::set<size_t>> &function_write_indices,
                const std::pair<std::vector<TypeId>, std::vector<bool>> &source_type_info,
                const std::pair<std::vector<TypeId>, std::vector<bool>> &target_type_info,
                std::vector<AnfNodePtr> *op_inputs) {
  auto func = function_write_indices.first;
  auto write_indices = function_write_indices.second;
  for (size_t i = 0; i < source_type_info.first.size(); ++i) {
    TypeId source_type_id = (source_type_info.first)[i];
    TypeId target_type_id = (target_type_info.first)[i];
    bool source_is_tensor = (source_type_info.second)[i];
    bool target_is_tensor = (target_type_info.second)[i];
    if (source_type_id == kTypeUnknown || target_type_id == kTypeUnknown) {
      continue;
    }
    if (source_type_id == target_type_id && source_is_tensor == target_is_tensor) {
      continue;
    }
    if (write_indices.find(i) != write_indices.end()) {
      RaiseExceptionForConvertRefDtype(func, TypeIdToString(source_type_id), TypeIdToString(target_type_id), i);
    }
    auto param = (*op_inputs)[i];
    auto target_type_node = NewValueNode(static_cast<int64_t>(target_type_id));
    MS_LOG(DEBUG) << "Do type cast for Primitive[" << func->ToString() << "], source_is_tensor: " << source_is_tensor
                  << ", target_is_tensor: " << target_is_tensor
                  << ", source_type_id: " << TypeIdToString(source_type_id)
                  << ", target_type_id: " << TypeIdToString(target_type_id);
    if (!source_is_tensor && target_is_tensor) {
      // Scalar needs to be converted to Tensor.
      auto source_type_node = NewValueNode(static_cast<int64_t>(source_type_id));
      param = func_graph->NewCNodeAfter(param, {NewValueNode(prim::kPrimScalarToTensor), param, source_type_node});
      (*op_inputs)[i] = func_graph->NewCNodeAfter(param, {NewValueNode(prim::kPrimCast), param, target_type_node});
    } else {
      // If target type is not Tensor but scalar, use ScalarCast.
      PrimitivePtr cast_op = target_is_tensor ? prim::kPrimCast : prim::kPrimScalarCast;
      (*op_inputs)[i] = func_graph->NewCNodeAfter(param, {NewValueNode(cast_op), param, target_type_node});
    }
  }
}
}  // namespace

std::vector<AnfNodePtr> GetNewInputsBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                                 const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                                 const std::vector<AnfNodePtr> &params_list) {
  // args: original inputs
  auto &signature = GetSignature(function);
  std::size_t sig_size = signature.size();
  auto has_var = (sig_size > 0 && signature[sig_size - 1].kind == SignatureEnumKind::kKindVarPositional);
  CheckSigSize(function, sig_size, has_var, args_abs_list, func_name);
  CheckPrimInputType(function, args_abs_list);
  std::vector<AnfNodePtr> op_inputs;
  std::set<size_t> write_indices;
  std::vector<TypePtr> input_types;
  auto cast_type = GetMixedPrecisionTargetType(func_graph);
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    AnfNodePtr param = params_list[i];
    if (args_abs_list[i] == nullptr) {
      op_inputs.push_back(param);
      continue;
    }

    SignatureEnumRW sig = GetSignatureEnumRW(i, signature, has_var);
    TypePtr type = args_abs_list[i]->BuildType();
    if (type && type->isa<RefType>()) {
      if (sig == SignatureEnumRW::kRWRead) {
        auto source_tensor_type = type->cast<TensorTypePtr>();
        if (source_tensor_type != nullptr) {
          auto source_element = source_tensor_type->element();
          if (cast_type != nullptr && (IsSubType(source_element, kFloat) || IsSubType(source_element, kBFloat)) &&
              *source_element != *cast_type) {
            auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
            param = func_graph->NewCNodeAfter(param, {NewValueNode(cast), param, NewValueNode(cast_type)});
            type = cast_type->type_id() == kNumberTypeFloat16
                     ? kTensorTypeFP16
                     : (cast_type->type_id() == kNumberTypeBFloat16 ? kTensorTypeBF16 : kTensorTypeFP32);
          }
        }
      } else if (sig == SignatureEnumRW::kRWWrite) {
        (void)write_indices.insert(i);
      }
      // If sig is SignatureEnumRW::kRWRef, not do anything.
    } else if (IsTypeRef(sig, type)) {
      RaiseExceptionForCheckParameter(func_name, i, type->ToString());
    }
    MS_LOG(DEBUG) << "Function " << func_name << "'s input " << i << " " << param->DebugString(2) << " abs "
                  << args_abs_list[i]->ToString() << " type " << type->ToString() << ".";
    input_types.push_back(type);
    op_inputs.push_back(param);
  }
  // process default
  ProcessDefault(func_name, args_abs_list.size(), signature, has_var, &op_inputs);
  // Record type info.
  std::vector<TypeId> source_type_id;
  std::vector<bool> source_is_tensor;
  GetTypeInfo(input_types, &source_type_id, &source_is_tensor);
  // Auto mixed precision.
  std::vector<TypeId> target_type_id = source_type_id;
  std::vector<bool> target_is_tensor = source_is_tensor;
  bool amp_type_changed = GetAutoMixedPrecisionType(
    func_graph, function, std::make_pair(source_type_id, source_is_tensor), &target_type_id, &target_is_tensor);
  // Implicit type promotion.
  bool promote_type_changed = GetImplicitPromoteType(signature, write_indices, &target_type_id, &target_is_tensor);
  // Do type cast.
  if (promote_type_changed || amp_type_changed) {
    DoTypeCast(func_graph, std::make_pair(function, write_indices), std::make_pair(source_type_id, source_is_tensor),
               std::make_pair(target_type_id, target_is_tensor), &op_inputs);
  }
  return op_inputs;
}

AnfNodePtr GenerateCNode(const FuncGraphPtr &func_graph, const std::string &func_name, const ValuePtr &function,
                         const AbstractBasePtrList &args_abs_list, const AnfNodePtrList &old_node_inputs) {
  auto new_inputs = GetNewInputsBySignatures(func_graph, func_name, function, args_abs_list, old_node_inputs);
  AnfNodePtrList op_inputs{NewValueNode(function)};
  (void)std::copy(new_inputs.begin(), new_inputs.end(), std::back_inserter(op_inputs));
  return func_graph->NewCNodeInOrder(op_inputs);
}

FuncGraphPtr DoSignatureMetaFuncGraph::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();

  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    (void)func_graph->add_parameter();
  }
  auto new_cnode = GenerateCNode(func_graph, name_, function_, args_abs_list, func_graph->parameters());
  func_graph->set_output(new_cnode);
  func_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  return func_graph;
}

void RaiseExceptionForConvertRefDtype(const ValuePtr &func, const std::string &ref_type, const std::string &target_type,
                                      size_t index) {
  std::ostringstream buffer;
  if (func->isa<Primitive>()) {
    auto prim = func->cast<PrimitivePtr>();
    auto args_names_value = prim->GetAttr("input_names");
    if (args_names_value != nullptr) {
      auto args_names = GetValue<std::vector<std::string>>(args_names_value);
      if (index < args_names.size()) {
        buffer << " the argument[" << args_names[index] << "]'s data type of primitive[" << prim->name() << "] is ";
      }
    }
  }
  if (buffer.str().empty()) {
    buffer << " so data type ";
  }
  MS_EXCEPTION(TypeError) << "Data type conversion of 'Parameter' is not supported," << buffer.str() << ref_type
                          << ", which cannot be converted to data type " << target_type << " automatically.\n";
}

void RaiseExceptionForCheckParameter(const std::string &func_name, size_t i, const std::string &source_type) {
  MS_EXCEPTION(TypeError) << "Function " << func_name << "'s input " << i << " should be a Parameter, but "
                          << source_type << ".";
}
}  // namespace prim
}  // namespace mindspore
