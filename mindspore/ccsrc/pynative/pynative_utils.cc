/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "pynative/pynative_utils.h"
#include <algorithm>
#include <vector>
#include <set>
#include "mindspore/ops/op_def/sparse_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "frontend/ir/primitive_py.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/primfunc_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/pynative/common_utils.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "include/common/utils/stub_tensor.h"
#include "frontend/expander/bprop/bprop.h"
#include "pynative/grad/jit/jit_grad.h"
#include "mindspore/ops/op_def/sequence_op_name.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "pynative/predict_out_type_map.h"
#include "mindspore/ccsrc/pyboost/auto_generate/contiguous.h"
#include "runtime/pipeline/pipeline.h"
#include "include/common/pynative/abstract_converter.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "pynative/grad/grad_utils.h"
#include "include/common/utils/tensor_py.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace pynative {
namespace PyNativeAlgo {
namespace {
std::string GetObjIdFromPython(const py::handle &obj) {
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}
// for simply infer (simple infer will push abs in bprop queue)
AbstractConverter kGradAbstractConverter;
const std::set<std::string> kVmOperators = {"InsertGradientOf", "StopGradient", "HookBackward", "CellBackwardHook"};
std::string GetIdForPyTupleOrList(const py::handle &obj) {
  auto p_list = py::cast<py::tuple>(obj);
  string prefix = py::isinstance<py::tuple>(obj) ? "Tuple<" : "List<";
  if (p_list.empty()) {
    prefix = "Empty:";
  } else {
    for (size_t i = 0; i < p_list.size(); ++i) {
      prefix += PyParser::GetIdByPyObj(p_list[i]) + ":";
    }
  }
  prefix.pop_back();
  prefix += ">";
  return prefix;
}

std::string GetFnInfoByPyObj(const py::object &obj) {
  std::string fn_info = obj.attr("__module__").cast<std::string>();
  fn_info += "_" + obj.attr("__name__").cast<std::string>();
  fn_info += "_" + obj.attr("__code__").attr("co_filename").cast<std::string>();
  fn_info += "_" + py::str(obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  if (py::hasattr(obj, "__warpped__")) {
    auto warpped_obj = obj.attr("__warpped__");
    fn_info += "_" + warpped_obj.attr("__name__").cast<std::string>();
    fn_info += "_" + warpped_obj.attr("__code__").attr("co_filename").cast<std::string>();
    fn_info += "_" + py::str(warpped_obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  }
  return fn_info;
}

void AddDynInputsSizesAttr(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->base_op_run_info.dyn_input_sizes.empty()) {
    return;
  }
  op_run_info->op_grad_info->op_prim->set_attr(kAttrDynInputSizes,
                                               MakeValue(op_run_info->base_op_run_info.dyn_input_sizes));
}

ValuePtr CreateNonTensorByAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto type_id = Common::GetTypeFromAbstract(abs);
  if (abs->isa<abstract::AbstractMonad>()) {
    return std::make_shared<tensor::Tensor>(0);
  }
  if (type_id == kMetaTypeNone) {
    return kNone;
  }
  if (type_id == kMetaTypeNull) {
    return kNull;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    ValuePtrList value_ptr_list;
    (void)std::transform(abs_seq.begin(), abs_seq.end(), std::back_inserter(value_ptr_list),
                         [](const abstract::AbstractBasePtr &elem) { return CreateNonTensorByAbstract(elem); });
    return std::make_shared<ValueTuple>(value_ptr_list);
  }
  if (type_id == kNumberTypeBool) {
    return MakeValue(true);
  }
  if (type_id == kObjectTypeString) {
    return MakeValue("");
  }
  if (type_id >= kNumberTypeInt && type_id <= kNumberTypeUInt64) {
    return MakeValue(static_cast<int64_t>(0));
  }
  if (type_id >= kNumberTypeFloat && type_id <= kNumberTypeFloat64) {
    return MakeValue(static_cast<float>(0));
  }
  if (type_id == kNumberTypeDouble) {
    return MakeValue(static_cast<double>(0));
  }
  MS_LOG(EXCEPTION) << "Get unsupported type " << type_id;
}

void PlantTupleParam(const FuncGraphPtr &bprop_graph, const abstract::AbstractSequencePtr &abs_seq,
                     AnfNodePtrList *make_tuple, AnfNodePtrList *new_param) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  MS_EXCEPTION_IF_NULL(make_tuple);
  MS_EXCEPTION_IF_NULL(new_param);
  MS_EXCEPTION_IF_NULL(abs_seq);
  for (size_t i = 0; i < abs_seq->size(); ++i) {
    const auto &cur_abs = abs_seq->elements()[i];
    if (cur_abs->isa<abstract::AbstractSequence>()) {
      auto is_tuple = cur_abs->isa<abstract::AbstractTuple>();
      AnfNodePtrList cur_make_tuple_inputs;
      auto prim = is_tuple ? prim::kPrimMakeTuple : prim::kPrimMakeList;
      (void)cur_make_tuple_inputs.emplace_back(NewValueNode(prim));
      PlantTupleParam(bprop_graph, cur_abs->cast<abstract::AbstractSequencePtr>(), &cur_make_tuple_inputs, new_param);
      auto cur_make_tuple_node = bprop_graph->NewCNode(cur_make_tuple_inputs);
      AbstractBasePtrList cur_abstract_elements;
      (void)std::transform(cur_make_tuple_inputs.begin() + 1, cur_make_tuple_inputs.end(),
                           std::back_inserter(cur_abstract_elements), [](const auto &e) { return e->abstract(); });
      AbstractBasePtr cur_abstract;
      if (is_tuple) {
        cur_abstract = std::make_shared<abstract::AbstractTuple>(cur_abstract_elements);
      } else {
        cur_abstract = std::make_shared<abstract::AbstractList>(cur_abstract_elements);
      }
      cur_make_tuple_node->set_abstract(cur_abstract);
      (void)make_tuple->emplace_back(cur_make_tuple_node);
    } else if (cur_abs->isa<abstract::AbstractTensor>()) {
      auto plant_param = bprop_graph->add_parameter();
      plant_param->set_abstract(cur_abs);
      (void)make_tuple->emplace_back(plant_param);
      (void)new_param->emplace_back(plant_param);
    } else if (cur_abs->isa<abstract::AbstractDictionary>()) {
      // Support output type of tuple contains dict
      const auto &abs_dict = cur_abs->cast<abstract::AbstractDictionaryPtr>();
      abstract::AbstractBasePtrList local_key_abs_inputs;
      abstract::AbstractBasePtrList local_value_abs_inputs;
      for (const auto &element : abs_dict->elements()) {
        (void)local_key_abs_inputs.emplace_back(element.first);
        (void)local_value_abs_inputs.emplace_back(element.second);
      }
      auto key_param = bprop_graph->add_parameter();
      key_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
      auto value_param = bprop_graph->add_parameter();
      value_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
      (void)new_param->emplace_back(key_param);
      (void)new_param->emplace_back(value_param);
      // Add Makedict node as tuple element
      auto dict_node = bprop_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), key_param, value_param});
      dict_node->set_abstract(abs_dict);
      (void)make_tuple->emplace_back(dict_node);
    } else {
      auto value = MakeValue(static_cast<int64_t>(0));
      auto value_node = NewValueNode(value);
      value_node->set_abstract(value->ToAbstract());
      (void)make_tuple->emplace_back(value_node);
    }
  }
}

const mindspore::HashSet<std::string> kNotRealOP{
  kMakeTupleOpName,
  kMakeListNewOpName,
  kTupleGetItemOpName,
  kStopGradientOpName,
  kUpdateStateOpName,
  kLoadOpName,
  kDependOpName,
  kReturnOpName,
  kShapeOpName,
  kNPUAllocFloatStatusOpName,
  kNPUGetFloatStatusOpName,
  kNPUClearFloatStatusOpName,
  kMirrorOperatorOpName,
  kSequenceSliceOpName,
  kSequenceMulOpName,
  kPyExecuteOpName,
};

tensor::BaseTensorPtr GetContiguousTensor(const tensor::BaseTensorPtr &input_tensor, const std::string &device_target,
                                          bool requires_grad) {
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_target);
  auto contiguous_tensor = contiguous_op->Call(input_tensor);
  if (requires_grad) {
    contiguous_op->CreateOutputSimpleInfo();
    const auto &contiguous_run_info = std::make_shared<FrontendOpRunInfo>();
    contiguous_run_info->requires_grad = true;
    auto real_output = AutoGradUtil::MakeOutput(
      true, contiguous_op, PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->top_cell()->op_index());
    PyNativeAlgo::AutoGradUtil::SetInferOutputToGrad(contiguous_run_info->op_grad_info, contiguous_op);
    PyBoost::UpdateStubOutput(contiguous_op, contiguous_run_info->stub_output, contiguous_op->output_abs(),
                              real_output);
    contiguous_run_info->base_op_run_info.device_target = device_target;
    contiguous_run_info->input_size = 1;
    contiguous_run_info->base_op_run_info.op_name = ops::kNameContiguous;
    contiguous_run_info->op_grad_info->op_prim = prim::kPrimContiguous;

    contiguous_run_info->op_grad_info->input_value = {input_tensor};
    contiguous_run_info->op_grad_info->out_value = real_output;
    PyBoost::DoGrad(contiguous_op, contiguous_run_info->op_grad_info, contiguous_run_info->async_status);
  }
  return contiguous_tensor;
}
}  // namespace

AnfNodePtr Common::ConvertValueSequenceToMakeTuple(const ValueNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &v = node->value();
  if (!v->isa<ValueSequence>()) {
    return node;
  }
  auto value_sequence = v->cast<ValueSequencePtr>();
  if (!node->abstract()->isa<abstract::AbstractSequence>() &&
      (node->abstract()->cast<abstract::AbstractSequencePtr>()->size() != value_sequence->size())) {
    MS_LOG(EXCEPTION) << "Get wrong matched abs " << node->abstract()->ToString() << " and value "
                      << value_sequence->ToString();
  }

  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &value : value_sequence->value()) {
    MS_EXCEPTION_IF_NULL(value);
    auto value_node = NewValueNode(value);
    auto abs = CommonUtils::SetAbstractValueToAnyValue(value->ToAbstract());
    value_node->set_abstract(abs);
    auto tuple_node = ConvertValueSequenceToMakeTuple(value_node, func_graph);
    (void)inputs.emplace_back(tuple_node);
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  auto make_tuple_node = func_graph->NewCNode(inputs);
  make_tuple_node->set_abstract(node->abstract());
  return make_tuple_node;
}

std::string Common::GetIdByValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>()->id();
  }
  if (v->isa<stub::StubNode>()) {
    return GetIdByValue(v->cast<stub::StubNodePtr>()->WaitValue());
  }
  if (v->isa<Cell>()) {
    return v->cast<CellPtr>()->id();
  }
  if (v->isa<mindspore::Type>()) {
    auto type_ptr = v->cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  }
  if (v->isa<StringImm>()) {
    return "S" + v->cast<StringImmPtr>()->value();
  }
  if (v->isa<BoolImm>()) {
    return "B" + std::to_string(v->cast<BoolImmPtr>()->value());
  }
  if (v->isa<IntegerImm>()) {
    return "I" + std::to_string(v->cast<Int64ImmPtr>()->value());
  }
  if (v->isa<FloatImm>()) {
    return "F" + std::to_string(v->cast<FP32ImmPtr>()->value());
  }
  if (v->isa<None>()) {
    return "None";
  }
  if (v->isa<Ellipsis>()) {
    return "Ellipsis";
  }
  if (v->isa<ValueSequence>()) {
    auto p_list = v->cast<ValueSequencePtr>();
    string prefix = v->isa<ValueTuple>() ? "Tuple<" : "List<";
    if (p_list->size() == 0) {
      prefix = "Empty:";
    } else {
      for (size_t i = 0; i < p_list->size(); ++i) {
        prefix += GetIdByValue(p_list->value()[i]) + ":";
      }
    }
    prefix.pop_back();
    prefix += ">";
    return prefix;
  }
  MS_LOG(DEBUG) << "Get type " << v->ToString();
  return v->ToString();
}

std::string Common::GetCellId(const std::string &obj_id, const std::vector<std::string> &input_arg_id_vec,
                              const std::vector<ValuePtr> &input_arg_value_vec) {
  auto cell_id = obj_id;
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  const auto &forward = GetPyNativeExecutor()->forward_executor();
  for (size_t i = 0; i < input_arg_id_vec.size(); ++i) {
    const auto &arg_id = input_arg_id_vec[i];
    // Find in step process
    auto cache_abs = forward->GetNodeAbsById(arg_id);
    if (cache_abs != nullptr) {
      fn(cache_abs);
    } else {
      MS_EXCEPTION_IF_NULL(input_arg_value_vec[i]);
      fn(CommonUtils::SetAbstractValueToAnyValue(input_arg_value_vec[i]->ToAbstract()));
    }
  }
  return cell_id;
}

void Common::SplitString(const std::string &str, std::vector<std::string> *id_vec) {
  constexpr char colon_delim = ':';
  constexpr char angle_bracket_left_delim = '<';
  constexpr char angle_bracket_right_delim = '>';
  auto paren_pos = str.find_first_of(angle_bracket_left_delim);
  if (paren_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Get wrong str " << str;
  }
  size_t str_size = str.size();
  const auto &sub_str = str.substr(paren_pos + 1, str_size - paren_pos - 2);
  MS_LOG(DEBUG) << "Ori str " << str << ", get sub str " << sub_str;
  size_t begin = 0;
  size_t angle_bracket_left = 0;
  size_t angle_bracket_right = 0;
  size_t sub_str_size = sub_str.size();
  for (size_t i = 0; i < sub_str_size; ++i) {
    switch (sub_str[i]) {
      case colon_delim:
        if (i != 0 && angle_bracket_left == angle_bracket_right) {
          (void)id_vec->emplace_back(sub_str.substr(begin, i - begin));
          begin = i + 1;
          angle_bracket_left = 0;
          angle_bracket_right = 0;
        }
        break;
      case angle_bracket_left_delim:
        ++angle_bracket_left;
        break;
      case angle_bracket_right_delim:
        ++angle_bracket_right;
        break;
      default: {
      }
    }
  }
  if (angle_bracket_left == angle_bracket_right) {
    (void)id_vec->emplace_back(sub_str.substr(begin, sub_str_size - begin));
  }
}

bool Common::ValueHasDynamicShape(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    return value->cast<tensor::BaseTensorPtr>()->base_shape_ptr() != nullptr;
  }
  if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return std::any_of(value_seq->value().begin(), value_seq->value().end(),
                       [](const ValuePtr &elem) { return ValueHasDynamicShape(elem); });
  }
  return false;
}

bool Common::IsTensor(const ValuePtr &v, bool include_sequence) {
  MS_EXCEPTION_IF_NULL(v);
  if (include_sequence) {
    if (v->isa<tensor::MetaSparseTensor>() || v->isa<tensor::BaseTensor>()) {
      return true;
    }
    if (v->isa<ValueSequence>()) {
      auto v_seq = v->cast<ValueSequencePtr>();
      if (v_seq->size() == 0) {
        MS_LOG(DEBUG) << "Get empty value sequence";
        return false;
      }
      // SpareTensor have scalar index, so just check have csr tensor
      if (v_seq->value().front()->isa<tensor::MetaSparseTensor>()) {
        return true;
      }
      // All value are tensor
      return std::all_of(v_seq->value().begin(), v_seq->value().end(),
                         [](const ValuePtr &e) { return IsTensor(e, true); });
    }
    MS_LOG(DEBUG) << "Get value " << v->ToString();
    return false;
  }
  MS_LOG(DEBUG) << "Get value " << v->ToString();
  return v->isa<tensor::BaseTensor>() || v->isa<tensor::MetaSparseTensor>();
}

bool Common::IsControlFlowGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

ValuePtr Common::FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>() || value->isa<tensor::COOTensor>() || value->isa<tensor::CSRTensor>()) {
    return value;
  }
  if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> value_list;
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (auto &filter_value : value_seq->value()) {
      if (auto t = FilterSensValues(filter_value, dict_convert_to_tuple); t != nullptr) {
        (void)value_list.emplace_back(t);
      }
    }
    return std::make_shared<ValueTuple>(value_list);
  }
  if (value->isa<ValueDictionary>()) {
    if (dict_convert_to_tuple) {
      return FilterSensValues(DataConvert::ConvertValueDictToValueTuple(value), dict_convert_to_tuple);
    }
    return value;
  }
  MS_LOG(DEBUG) << "Value type: " << value->ToString();
  return nullptr;
}

tensor::BaseTensorPtr Common::GetTensorFromParam(const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(param_node);
  auto param = param_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  if (!param->has_default()) {
    return nullptr;
  }
  auto default_value = param->default_param();
  MS_EXCEPTION_IF_NULL(default_value);
  auto tensor_value = default_value->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_value);
  return tensor_value;
}

const std::shared_ptr<PyNativeExecutor> &Common::GetPyNativeExecutor() {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  return executor;
}

TypeId Common::GetTypeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    return GetTypeFromAbstract(abs_seq->elements().front());
  }
  const auto &type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  return common::AnfAlgo::GetOutputInferDataType(type, 0);
}

ShapeVector Common::GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get abstract sequence";
  }
  auto shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_ptr = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr->shape();
}

std::pair<TypePtr, TypeId> Common::GetTypeFromValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    return std::make_pair(v->cast<tensor::BaseTensorPtr>()->Dtype(), kObjectTypeTensorType);
  }
  if (v->isa<ValueTuple>()) {
    return std::make_pair(v->type(), kObjectTypeTuple);
  }
  if (v->isa<ValueList>()) {
    return std::make_pair(v->type(), kObjectTypeList);
  }
  if (v->isa<None>()) {
    return std::make_pair(kTypeNone, kMetaTypeNone);
  }
  return std::make_pair(v->type(), v->type()->object_type());
}

ShapeVector Common::GetShapeFromValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>()->shape_c();
  }
  if (v->isa<ValueSequence>()) {
    const auto &v_seq = v->cast<ValueSequencePtr>()->value();
    ShapeVector plant_shape_vector;
    for (const auto &item : v_seq) {
      const auto &shape = GetShapeFromValue(item);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(plant_shape_vector),
                           [](int64_t s) { return s; });
    }
    return plant_shape_vector;
  }
  return ShapeVector{};
}

ValuePtr Common::CreatOutputTensorValueByAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto type_id = GetTypeFromAbstract(abs);
  if (abs->isa<abstract::AbstractMonad>()) {
    return std::make_shared<tensor::Tensor>(0);
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> out;
    if (!abs_seq->elements().front()->isa<abstract::AbstractTensor>()) {
      MS_LOG(DEBUG) << "Get non tensor output";
      return CreateNonTensorByAbstract(abs);
    }
    for (size_t i = 0; i < abs_seq->size(); ++i) {
      (void)out.emplace_back(std::make_shared<tensor::Tensor>(type_id, GetShapeFromAbstract(abs_seq->elements()[i])));
    }
    return std::make_shared<ValueTuple>(out);
  }
  if (!abs->isa<abstract::AbstractTensor>()) {
    MS_LOG(DEBUG) << "Get non tensor output";
    return CreateNonTensorByAbstract(abs);
  }
  return std::make_shared<tensor::Tensor>(type_id, GetShapeFromAbstract(abs));
}

void Common::ReplaceCNodeWithValueNode(const FuncGraphPtr &bprop_graph) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  if (bprop_graph->used_forward_nodes().empty()) {
    return;
  }
  auto mng = MakeManager({bprop_graph}, false);
  auto tr = mng->Transact();
  for (const auto &forward_node : bprop_graph->used_forward_nodes()) {
    auto cnode = forward_node->cast<CNodePtr>();
    auto v_node = cnode->forward().first;
    MS_EXCEPTION_IF_NULL(v_node);
    bprop_graph->AddValueNode(v_node);
    MS_LOG(DEBUG) << "Replace " << forward_node->DebugString() << " by value node " << v_node->DebugString();
    auto converted_node = ConvertValueSequenceToMakeTuple(v_node, bprop_graph);
    (void)tr.Replace(forward_node, converted_node);
  }
  tr.Commit();
  bprop_graph->ClearUsedForwardNodes();
  CommonUtils::DumpGraphIR("replace_cnode_with_valuenode.ir", bprop_graph);
}

ValuePtr Common::StubNodeToValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue();
  }
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    bool has_stub =
      std::any_of(values.begin(), values.end(), [](const auto &v) { return utils::isa<stub::StubNode>(v); });
    if (!has_stub) {
      return v;
    }
    ValuePtrList value_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(value_list),
                         [](const ValuePtr &value) { return StubNodeToValue(value); });
    if (utils::isa<ValueTuple>(v)) {
      return std::make_shared<ValueTuple>(value_list);
    }
    if (utils::isa<ValueList>(v)) {
      return std::make_shared<ValueList>(value_list);
    }
    MS_LOG(EXCEPTION) << "Value not support ValueSequence " << v->ToString();
  } else {
    return v;
  }
}

void Common::StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info);
  auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
  kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->base_op_run_info.stream_id);
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    op_run_info->op_grad_info->input_value[i] = StubNodeToValue(op_run_info->op_grad_info->input_value[i]);
    // Contiguous tensor in Backend RunOp.
    kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);
    runtime::DeviceAddressUtils::CreateKernelTensor(op_run_info->op_grad_info->input_value[i]);
  }
}

tensor::BaseTensorPtr Common::StubNodeToTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue()->cast<tensor::BaseTensorPtr>();
  }
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>();
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor, but got " << v->ToString();
}

ValuePtr Common::ConvertToContiguousValue(const ValuePtr &v, bool requires_grad) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->storage_info() == nullptr) {
      return tensor;
    }

    auto contiguous_tensor = ConvertToContiguousTensor(tensor, requires_grad);
    MS_LOG(DEBUG) << "ConvertToContiguousValue, old tensor id:" << tensor->id()
                  << ", new tensor id:" << contiguous_tensor->id();
    return contiguous_tensor;
  }
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    if (values.empty() || utils::isa<Scalar>(values[0])) {
      return v;
    }
    ValuePtrList value_list;
    (void)std::transform(
      values.begin(), values.end(), std::back_inserter(value_list),
      [requires_grad](const ValuePtr &value) { return ConvertToContiguousValue(value, requires_grad); });
    if (utils::isa<ValueTuple>(v)) {
      return std::make_shared<ValueTuple>(value_list);
    }
    if (utils::isa<ValueList>(v)) {
      return std::make_shared<ValueList>(value_list);
    }
    MS_LOG(EXCEPTION) << "Not support ValueSequence " << v->ToString();
  } else {
    return v;
  }
}

tensor::BaseTensorPtr Common::ConvertToContiguousTensor(const tensor::BaseTensorPtr &tensor, bool requires_grad) {
  MS_EXCEPTION_IF_NULL(tensor);

  // Tensor with storage info, need convert to contiguous in no-view op.
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  const auto &device_target = device_address->device_name();

  return GetContiguousTensor(tensor, device_target, requires_grad);
}

tensor::BaseTensorPtr Common::ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous, bool requires_grad) {
  const auto &tensor = StubNodeToTensor(v);
  MS_EXCEPTION_IF_NULL(tensor);
  if (!need_contiguous || tensor->storage_info() == nullptr) {
    return tensor;
  }

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  const auto &device_target = device_address->device_name();
  if (device_target == kAscendDevice) {
    return tensor;
  }

  return GetContiguousTensor(tensor, device_target, requires_grad);
}

std::optional<tensor::BaseTensorPtr> Common::ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                     bool need_contiguous, bool requires_grad) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToTensor(v.value(), need_contiguous, requires_grad));
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

std::optional<ValueTuplePtr> Common::ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                 bool need_contiguous, bool requires_grad) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToValueTuple(v.value(), need_contiguous, requires_grad));
}

void Common::GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                 const std::string &device_target, bool is_dynamic_shape,
                                 mindspore::HashSet<size_t> *input_to_attr_index) {
  if (op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    PrimitiveReadLock read_lock(op_prim->shared_mutex());
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_index);
    return;
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return;
  }

  auto reg_info =
    opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, device_target, is_dynamic_shape);
  if (reg_info == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(input_to_attr_index);
  for (auto &iter : reg_info->input_attr_map()) {
    (void)input_to_attr_index->insert(iter.first);
  }
}

ValueNodePtr Common::CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  auto v_node = NewValueNode(v);
  if (abs == nullptr) {
    v_node->set_abstract(CommonUtils::SetAbstractValueToAnyValue(v->ToAbstract()));
  } else {
    v_node->set_abstract(abs);
  }
  return v_node;
}

tensor::TensorPtr Common::CreateFakeTensorWithoutDeviceAddress(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto t = std::make_shared<tensor::Tensor>(*tensor);
  if (tensor->is_parameter()) {
    t->set_param_info(tensor->param_info());
  }
  t->set_device_address(nullptr);
  return t;
}

void Common::ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::BaseTensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (const auto &tensor : tensors) {
    tensor->set_device_address(nullptr);
  }
}

void Common::SetOutputUsedInBpropGraph(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &v_t = value->cast<tensor::BaseTensorPtr>();
    v_t->set_used_in_bprop_graph(true);
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (const auto &v : value_seq->value()) {
      SetOutputUsedInBpropGraph(v);
    }
  }
  if (value->isa<stub::StubNode>()) {
    const auto &stub_node = value->cast<stub::StubNodePtr>();
    return SetOutputUsedInBpropGraph(stub_node->WaitValue());
  }
  if (value->isa<ValueDictionary>()) {
    auto dic_v = value->cast<ValueDictionaryPtr>();
    for (const auto &v : dic_v->value()) {
      SetOutputUsedInBpropGraph(v.second);
    }
  }
}

ValuePtr Common::CreateFakeValueWithoutDeviceAddress(const ValuePtr &value, bool is_force_create_fake) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &v_t = value->cast<tensor::BaseTensorPtr>();
    // If the tensor used in bprop graph, no need create fake value
    if (!is_force_create_fake && v_t->is_parameter()) {
      return value;
    }

    auto t = std::make_shared<tensor::BaseTensor>(*v_t);
    if (v_t->is_parameter()) {
      t->set_param_info(v_t->param_info());
    }
    // For view tensor, we need keep storage info for do grad.
    if (v_t->storage_info() != nullptr) {
      t->set_storage_info(v_t->storage_info());
    }
    t->set_device_address(nullptr);
    t->set_used_in_bprop_graph(false);
    return t;
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    ValuePtrList value_list;
    (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(value_list),
                         [](const ValuePtr &elem) { return CreateFakeValueWithoutDeviceAddress(elem); });
    return std::make_shared<ValueTuple>(value_list);
  }
  if (value->isa<stub::StubNode>()) {
    const auto &stub_node = value->cast<stub::StubNodePtr>();
    return CreateFakeValueWithoutDeviceAddress(stub_node->WaitValue());
  }
  if (value->isa<ValueDictionary>()) {
    auto dic_v = value->cast<ValueDictionaryPtr>();
    std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
    for (const auto &v : dic_v->value()) {
      (void)key_values.emplace_back(v.first, CreateFakeValueWithoutDeviceAddress(v.second));
    }
    return std::make_shared<ValueDictionary>(key_values);
  }
  return value;
}

void Common::SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &original_params = func_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  bool need_add_input_abs = op_run_info->op_grad_info->input_abs.empty();
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      op_run_info->op_grad_info->input_value_grad_type[i] =
        AutoGradUtil::SetValueGradInfo(op_run_info->op_grad_info->input_value[i], InputType::kConstant);
      if (need_add_input_abs) {
        (void)op_run_info->op_grad_info->input_abs.emplace_back(original_params[i]->abstract());
      }
      continue;
    }
    // Must weight param
    // Parameters current used in inner graph, and no used in outer graph
    const auto &param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    (void)op_run_info->op_grad_info->input_value.emplace_back(tensor_value);
    (void)op_run_info->op_grad_info->input_value_grad_type.emplace_back(AutoGradUtil::SetTensorGradInfo(tensor_value));
    (void)op_run_info->op_grad_info->input_abs.emplace_back(param->abstract());
    MS_LOG(DEBUG) << "Set graph weight parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

void Common::FreeFuncGraphForwardNodes(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->used_forward_nodes().empty()) {
    return;
  }
  for (const auto &node : func_graph->used_forward_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_forward(nullptr, "");
  }
  func_graph->ClearUsedForwardNodes();
}

ValuePtr Common::CreateTensorByConstantValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->type();
  if (Common::IsTensor(value, true) || value->isa<Number>() || value->isa<None>() ||
      (type != nullptr && type->isa<String>())) {
    return value;
  }
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else if (value->isa<ValueList>()) {
    tensor_ptr = opt::CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value()));
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple, but get type " << value->type_name()
                      << ", value " << value->ToString();
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  return tensor_ptr;
}

bool Common::IsHookNeedSaveInputs(const PrimitivePyPtr &prim) {
  if (prim->hook_type() == HookType::kCustomOpBprop || prim->hook_type() == HookType::kCellCustomBprop) {
    return true;
  }
  return false;
}

bool Common::IsVmOp(const std::string &op_name) { return kVmOperators.find(op_name) != kVmOperators.end(); }

std::vector<int64_t> Common::BuildShape(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->BuildShape();
  if (base_shape->isa<abstract::NoShape>()) {
    return {};
  }
  auto shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape->shape();
}

void Common::ClearRes() { kGradAbstractConverter.clear(); }

std::string PyParser::GetIdByPyObj(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return tensor::ConvertToBaseTensor(obj)->id();
  }
  if (IsStubTensor(obj)) {
    return ConvertStubTensor(obj)->id();
  }
  if (py::isinstance<Cell>(obj)) {
    return obj.cast<CellPtr>()->id();
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = obj.cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  }
  if (py::isinstance<py::str>(obj)) {
    return "S" + obj.cast<std::string>();
  }
  if (py::isinstance<py::bool_>(obj)) {
    return "B" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::int_>(obj)) {
    return "I" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::float_>(obj)) {
    return "F" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::none>(obj)) {
    return "None";
  }
  if (py::isinstance<py::ellipsis>(obj)) {
    return "Ellipsis";
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    return GetIdForPyTupleOrList(obj);
  }
  if (py::isinstance<py::function>(obj)) {
    return GetFnInfoByPyObj(obj);
  }
  // For id with value and obj can be the same
  if (py::isinstance<tensor::CSRTensor>(obj) || py::isinstance<tensor::COOTensor>(obj) ||
      py::isinstance<tensor::RowTensor>(obj)) {
    return parse::data_converter::PyObjToValue(obj)->ToString();
  }
  return GetObjIdFromPython(obj);
}

std::pair<std::vector<std::string>, std::vector<ValuePtr>> PyParser::GetArgsIdAndValue(const py::args &args) {
  size_t arg_size = args.size();
  std::vector<std::string> input_arg_id_vec;
  std::vector<ValuePtr> input_arg_value_vec;
  input_arg_id_vec.reserve(arg_size);
  input_arg_value_vec.reserve(arg_size);
  for (size_t i = 0; i < arg_size; ++i) {
    if (py::isinstance<py::list>(args[i])) {
      (void)input_arg_value_vec.emplace_back(parse::data_converter::PyObjToValue(py::cast<py::tuple>(args[i])));
    } else {
      (void)input_arg_value_vec.emplace_back(parse::data_converter::PyObjToValue(args[i]));
    }
    (void)input_arg_id_vec.emplace_back(Common::GetIdByValue(input_arg_value_vec.back()));
  }
  return {input_arg_id_vec, input_arg_value_vec};
}

void PyParser::SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &adapter = prim_arg.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(prim_arg);
    adapter->set_attached_primitive(prim);
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  op_run_info->signatures = prim->signatures();
  op_run_info->base_op_run_info.py_prim_id_ = adapter->id();
}

std::string PyParser::BuilidPyInputTypeString(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return "Tensor";
  }
  if (IsStubTensor(obj)) {
    return "Tensor";
  }
  // bool must before int, because bool is a special int
  if (py::isinstance<py::bool_>(obj)) {
    return "bool";
  }
  if (py::isinstance<py::int_>(obj)) {
    return "int";
  }
  if (py::isinstance<py::float_>(obj)) {
    return "float";
  }
  if (py::isinstance<py::str>(obj)) {
    return "string";
  }
  if (py::isinstance<py::none>(obj)) {
    return "None";
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    return "mindspore.dtype";
  }

  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    std::stringstream ss;
    ss << (py::isinstance<py::tuple>(obj) ? "Tuple<" : "List<");
    auto tuple = py::cast<py::tuple>(obj);
    for (size_t i = 0; i < tuple.size(); i++) {
      if (i == 0) {
        ss << BuilidPyInputTypeString(tuple[i]);
      } else {
        ss << ", " << BuilidPyInputTypeString(tuple[i]);
      }
    }
    ss << ">";
    return ss.str();
  }

  std::stringstream ss;
  ss << obj.get_type();
  return ss.str();
}

void PyParser::PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx) {
  auto const &op_arg = op_def->args_[idx];
  bool is_suppport_tensor_cast = std::any_of(op_arg.cast_dtype_.begin(), op_arg.cast_dtype_.end(),
                                             [](const auto &type) { return type == ops::DT_TENSOR; });
  if (is_suppport_tensor_cast) {
    auto tensor = parse::ConvertTensorValue(op_inputs[idx]);
    auto PrintVectorFunc = [](const ShapeVector &shape) -> std::string {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) {
          ss << ", " << shape[i];
        } else {
          ss << shape[i];
        }
      }
      ss << "]";
      return ss.str();
    };
    if (tensor != nullptr) {
      MS_EXCEPTION(TypeError) << "For " << op_def->name_ << ", the " << idx << "'th input is a Tensor whose shape is "
                              << PrintVectorFunc(tensor->shape()) << " and dtype is ["
                              << TypeIdToString(tensor->data_type()) << "], which can not be converted to "
                              << ops::EnumToString(op_arg.arg_dtype_) << ".";
    }
  }
  std::vector<std::string> op_type_list;
  for (size_t index = 0; index < op_inputs.size(); ++index) {
    (void)op_type_list.emplace_back(BuilidPyInputTypeString(op_inputs[index]));
  }
  PyNativeExecutor::GetInstance()->ClearRes();
  MS_EXCEPTION(TypeError) << ops::BuildOpErrorMsg(op_def, op_type_list);
}

inline ValuePtr ConvertScalarToTensor(const ValuePtr &value) {
  auto fp32_imm = value->cast<FP32ImmPtr>();
  if (fp32_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(fp32_imm->value());
  }

  auto bool_imm = value->cast<BoolImmPtr>();
  if (bool_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(bool_imm->value());
  }

  auto int64_imm = value->cast<Int64ImmPtr>();
  if (int64_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(int64_imm->value());
  }

  MS_LOG(EXCEPTION) << "Unsupported type: " << value->ToString();
}

inline ValuePtr ConvertBySignature(const py::object &obj, const FrontendOpRunInfoPtr &op_run_info, size_t index) {
  if (op_run_info->signatures.size() <= index) {
    return nullptr;
  }

  if (op_run_info->signatures[index].dtype != SignatureEnumDType::kDTypeEmptyDefaultValue) {
    auto convert_func = parse::GetConverterByType(static_cast<int32_t>(ops::DT_NUMBER));
    MS_EXCEPTION_IF_NULL(convert_func);
    return convert_func(obj);
  }
  return nullptr;
}

void ParseOpInputByOpDef(const ops::OpDefPtr &op_def, const py::list &op_inputs, bool stub,
                         const FrontendOpRunInfoPtr &op_run_info) {
  size_t input_size = op_inputs.size();
  if (input_size != op_def->args_.size()) {
    MS_LOG(EXCEPTION) << "For Operator[" << op_def->name_ << "], the inputs number should be " << op_def->args_.size()
                      << " but got " << op_inputs.size() << ".";
  }
  (void)op_run_info->op_grad_info->input_value.resize(input_size);
  for (size_t i = 0; i < op_def->args_.size(); i++) {
    auto const &op_arg = op_def->args_[i];
    op_run_info->none_init_inputs_num += static_cast<size_t>(!op_arg.as_init_arg_);

    // Optional argument is valid for None as input.
    if (op_arg.is_optional_ && py::isinstance<py::none>(op_inputs[i])) {
      op_run_info->op_grad_info->input_value[i] = kNone;
      continue;
    }

    ValuePtr value = nullptr;
    parse::OpDefConvertFunc convert_func = parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
    MS_EXCEPTION_IF_NULL(convert_func);
    value = convert_func(op_inputs[i]);
    if (value != nullptr) {
      op_run_info->op_grad_info->input_value[i] = value;
      continue;
    }

    // type cast has lower priority then signature cast
    if (!op_arg.cast_dtype_.empty()) {
      for (auto cast_dtype : op_arg.cast_dtype_) {
        convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
        MS_EXCEPTION_IF_NULL(convert_func);
        value = convert_func(op_inputs[i]);
        if (value != nullptr) {
          op_run_info->op_grad_info->input_value[i] = value;
          op_run_info->source_type[i] = cast_dtype;
          break;
        }
      }
    }

    if (value == nullptr) {
      PyParser::PrintTypeCastError(op_def, op_inputs, i);
    }
  }
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs, bool stub) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  op_run_info->source_type.resize(op_run_info->input_size);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);

  auto op_def = mindspore::ops::GetOpDef(op_run_info->base_op_run_info.op_name);
  if (op_def == nullptr) {
    op_run_info->op_grad_info->input_value.resize(op_run_info->input_size);
    op_run_info->none_init_inputs_num = op_run_info->input_size;
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      op_run_info->op_grad_info->input_value[i] = parse::data_converter::PyObjToValue(op_inputs[i], stub);
    }
  } else {
    op_run_info->none_init_inputs_num = 0;
    ParseOpInputByOpDef(op_def, op_inputs, stub, op_run_info);
  }
}

ValuePtrList DataConvert::FlattenOnlyTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList outputs;
  CommonUtils::FlattenValueSeqArg(v, false, true, &outputs);
  return outputs;
}

ValuePtrList DataConvert::FlattenTensorSeqInValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList outputs;
  CommonUtils::FlattenValueSeqArg(v, true, false, &outputs);
  return outputs;
}

void DataConvert::FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v, bool has_sens) {
  MS_EXCEPTION_IF_NULL(flatten_v);
  if (v_vec.empty()) {
    MS_LOG(EXCEPTION) << "For bprop graph input value size should be greatet than 0, but get empty.";
  }
  size_t input_size = has_sens ? v_vec.size() - 1 : v_vec.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto &v = v_vec[i];
    MS_EXCEPTION_IF_NULL(v);
    MS_LOG(DEBUG) << "Get v is " << v->ToString();
    (void)flatten_v->emplace_back(v);
  }
  if (has_sens) {
    if (Common::IsTensor(v_vec[input_size])) {
      (void)flatten_v->emplace_back(v_vec[input_size]);
    } else if (v_vec[input_size]->isa<ValueSequence>()) {
      MS_LOG(DEBUG) << "Get value tuple size " << v_vec[input_size]->cast<ValueSequencePtr>()->size();
      CommonUtils::FlattenValueSeqArg(v_vec[input_size], false, false, flatten_v);
    }
  }
}

bool DataConvert::RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                               size_t input_index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->input_to_attr.empty()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(v);
  if (op_run_info->input_to_attr.find(input_index) == op_run_info->input_to_attr.end()) {
    return false;
  }
  const auto &input_names_value = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return false;
  }
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "The input index: " << input_index << " is larger than the input names vector size!";
  }
  const auto &input_name = input_names_vec[input_index];
  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
      return false;
    }
  }
  (void)op_run_info->op_grad_info->op_prim->AddAttr(input_name, v);
  return true;
}

void DataConvert::TransformValueNodeBaseTensorToTensor(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &v = value_node->value();
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<tensor::BaseTensor>()) {
    return;
  }
  const auto &tensor = v->cast<tensor::BaseTensorPtr>();
  value_node->set_value(std::make_shared<tensor::Tensor>(*tensor));
}

ValuePtr DataConvert::ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs) {
  if (values.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "tensors size should not be empty!";
  }
  if (values.size() == kSizeOne && !abs->isa<abstract::AbstractSequence>()) {
    return values[kIndex0];
  }
  return std::make_shared<ValueTuple>(values);
}

ValuePtrList DataConvert::TensorListToValueList(const tensor::BaseTensorPtrList &tensor_list) {
  ValuePtrList output_values;
  output_values.reserve(tensor_list.size());
  (void)std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(output_values),
                       [](const BaseTensorPtr &tensor) -> ValuePtr {
                         if (tensor == nullptr) return kNone;
                         return tensor;
                       });
  return output_values;
}

FrontendOpRunInfoPtr PyBoost::Init(const PrimitivePtr &prim) {
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  pynative_executor->StoreAsyncStatus(op_run_info);
  forward_executor->InitOpRunInfo(op_run_info);
  return op_run_info;
}

void PyBoost::UpdateStubOutput(const kernel::pyboost::OpPtr &op, const stub::StubNodePtr &stub_output,
                               const AbstractBasePtr &abstract, const ValuePtr &real_out) {
  MS_EXCEPTION_IF_NULL(op);
  if (stub_output == nullptr || stub_output->isa<stub::NoneTypeNode>()) {
    return;
  }
  if (MS_UNLIKELY(op->output_value_simple_info() != nullptr)) {
    stub_output->SetValueSimpleInfo(op->output_value_simple_info());
  } else {
    MS_EXCEPTION_IF_NULL(abstract);
    auto success = stub_output->SetAbstract(abstract);
    if (!success) {
      MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                              << PredictOutTypeByName(op->primitive()->name()) << ", infer type is "
                              << abstract->BuildType() << ", the name of operator is [" << op->primitive()->name()
                              << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
    }
    MS_LOG(DEBUG) << "Update StubNode abstract " << abstract->ToString();
  }
  stub_output->SetValue(real_out);
}

void PyBoost::DataSyncForGraph(const kernel::pyboost::OpPtr &op) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !runtime::OpExecutor::GetInstance().async_for_graph()) {
    // If execution mode is Graph Mode in MsContext, the tensor will be the input of graph which will execute in Graph
    // Mode, if the graph contain no CNode after optimization, the tensor need sync to host.
    for (const auto &output : op->outputs()) {
      auto device_address = std::static_pointer_cast<device::DeviceAddress>(output->device_address());
      if (device_address == nullptr) {
        continue;
      }
      runtime::DeviceAddressUtils::CreateKernelTensor(device_address, output.get());
      output->data_sync(true);
    }
  }
}

PrimitivePtr PyBoost::ConvertPrimitive(const py::object &obj) {
  const auto &adapter = obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);

  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
#ifndef ENABLE_TEST
    // Custom operator's infer type and backpropagation are defined on the Python side.
    if (adapter->name() != kCustomExtOpName) {
      return std::make_shared<Primitive>(adapter->name(), adapter->attrs());
    }
    prim = std::make_shared<PrimitivePy>(obj);
    adapter->set_attached_primitive(prim);
#else
    prim = std::make_shared<PrimitivePy>(obj);
    adapter->set_attached_primitive(prim);
#endif
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  prim->EnableSharedMutex();
  return prim;
}

py::object PyBoost::RunPyFunction(const PrimitivePtr &prim, const py::list &args) {
  py::tuple wrap_args(kIndex3);
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (!prim_py->HasPyObj()) {
      MS_LOG(EXCEPTION) << "Prim has not python obj!";
    }
    wrap_args[kIndex0] = prim_py->GetPyObj();
  } else {
    wrap_args[kIndex0] = std::make_shared<PrimitivePyAdapter>(prim->name());
  }
  wrap_args[kIndex1] = prim->name();
  wrap_args[kIndex2] = args;
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  return pynative_executor->RunOpStub(wrap_args);
}

void PyBoost::SetAnyValueForAbstract(const kernel::pyboost::OpPtr &op) {
  const auto &input_abs = op->input_abs();
  for (const auto &abs : input_abs) {
    CommonUtils::SetAbstractValueToAnyValue(abs);
  }
  CommonUtils::SetAbstractValueToAnyValue(op->output_abs());
}

void PyBoost::DoGrad(const kernel::pyboost::OpPtr &op, const OpGradInfoPtr &grad_info,
                     const AsyncStatus &async_status) {
  static const std::string kDoGradName = "DoGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);

  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();
  if (op->output_value_simple_info() == nullptr) {
    MS_LOG(EXCEPTION) << "The simple info of " << op->primitive()->name() << " infer is null";
  }
  // Inplace op need save clone tensor.
  grad_info->clone_value = op->clone_tensor();
  // Check and set input auto grad meta info and InputType
  if (MS_LIKELY(!forward->grad()->top_cell()->is_bprop_need_get_forward_graph())) {
    MarkPyBoostInputs(grad_info, forward->grad()->top_cell());
  }
  forward->ForwardOpGradImpl(grad_info, async_status);
}

void PyBoost::MarkPyBoostInputs(const OpGradInfoPtr &op_grad_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  size_t input_size = op_grad_info->input_value.size();
  op_grad_info->input_value_grad_type.resize(input_size);
  for (size_t index = 0; index < input_size; ++index) {
    const auto &v = op_grad_info->input_value[index];
    if (v->isa<tensor::BaseTensor>()) {
      op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(v->cast<tensor::BaseTensorPtr>());
    } else if (v->isa<ValueSequence>()) {
      const auto &value_sequence = v->cast<ValueSequencePtr>();
      const auto &tuple_inputs = value_sequence->value();
      if (!tuple_inputs.empty() && tuple_inputs[0]->isa<tensor::BaseTensor>()) {
        op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
        for (const auto &elem : tuple_inputs) {
          auto grad_type = AutoGradUtil::SetTensorGradInfo(elem->cast<tensor::BaseTensorPtr>());
          if (AutoGradUtil::IsParam(grad_type)) {
            op_grad_info->input_value_grad_type[index] = InputType::kParameter;
          }
        }
      }
    } else if (v->isa<tensor::MapTensor>()) {
      op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(v->cast<tensor::MapTensorPtr>());
    } else if (v->isa<tensor::CSRTensor>()) {
      const auto &csr_tensor = v->cast<tensor::CSRTensorPtr>();
      auto fn = [&op_grad_info, index](const auto &csr_tensor_input) {
        auto grad_type = AutoGradUtil::SetTensorGradInfo(csr_tensor_input);
        if (AutoGradUtil::IsParam(grad_type)) {
          op_grad_info->input_value_grad_type[index] = InputType::kParameter;
        }
      };
      op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
      fn(csr_tensor->GetIndptr());
      fn(csr_tensor->GetIndices());
      fn(csr_tensor->GetValues());
    }
  }
}

void PyBoost::BumpVersionAsync(const tensor::BaseTensorPtr &tensor) {
  const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
  if (forward->enable_async()) {
    const auto task = [tensor]() { tensor->BumpVersion(); };
    const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
    bprop_queue->Push(std::make_shared<BpropTask>(task));
  } else {
    tensor->BumpVersion();
  }
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
  }
  for (const auto &v : value_seq->value()) {
    if (!v->isa<tensor::BaseTensor>()) {
      MS_LOG(DEBUG) << "Get value " << v->ToString() << " in tensor tuple, op name "
                    << op_run_info->base_op_run_info.op_name;
      (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
      (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
      continue;
    }
    InputType input_type = InputType::kInput;
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      auto grad_type = AutoGradUtil::SetTensorGradInfo(tensor);
      if (AutoGradUtil::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kParameter;
      }
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(input_type);
  }

  if (!op_run_info->base_op_run_info.dyn_input_sizes.empty()) {
    int64_t elem_size = SizeToLong(value_seq->size());
    if (op_run_info->base_op_run_info.dyn_input_sizes.size() != op_run_info->input_size) {
      for (size_t i = op_run_info->base_op_run_info.dyn_input_sizes.size(); i < index; ++i) {
        (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
      }
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(elem_size);
    } else {
      op_run_info->base_op_run_info.dyn_input_sizes[index] = elem_size;
    }
  } else {
    for (size_t i = 0; i < index; ++i) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
    (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(SizeToLong(value_seq->size()));
  }
}

ValuePtr DataConvert::ConvertValueDictToValueTuple(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  const auto &dic_v = v->cast<ValueDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(dic_v);
  std::vector<ValuePtr> v_list;
  (void)std::transform(dic_v->value().begin(), dic_v->value().end(), std::back_inserter(v_list),
                       [](const std::pair<ValuePtr, ValuePtr> &elem) { return elem.second; });
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                                   const TopCellInfoPtr &top_cell, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(map_tensor);
  constexpr int input_num = 1;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(map_tensor);
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kParameter);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(map_tensor);
  }
}

void DataConvert::ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                               const tensor::CSRTensorPtr &csr_tensor, const TopCellInfoPtr &top_cell,
                                               size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(csr_tensor);
  constexpr int input_num = 3;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }

  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kInput);
  op_run_info->op_grad_info->op_prim->set_attr("is_csr", MakeValue(true));
  op_run_info->op_grad_info->op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
    for (int i = 0; i < input_num; ++i) {
      auto iter = op_run_info->base_op_run_info.expanded_input_values.rbegin() + i;
      auto grad_type = AutoGradUtil::SetTensorGradInfo((*iter)->cast<tensor::BaseTensorPtr>());
      if (AutoGradUtil::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kParameter;
      }
    }
  }
}

void DataConvert::GetTensorIdFromOutputValue(const ValuePtr &value, std::vector<std::string> *converted_tensor_id) {
  if (value->isa<tensor::BaseTensor>()) {
    (void)converted_tensor_id->emplace_back(value->cast<tensor::BaseTensorPtr>()->id());
    MS_LOG(DEBUG) << "Get top cell output tensor id " << converted_tensor_id->back();
  } else if (value->isa<ValueSequence>()) {
    const auto &seq = value->cast<ValueSequencePtr>();
    for (const auto &val : seq->value()) {
      GetTensorIdFromOutputValue(val, converted_tensor_id);
    }
  } else if (value->isa<ValueDictionary>()) {
    GetTensorIdFromOutputValue(ConvertValueDictToValueTuple(value), converted_tensor_id);
  }
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::BaseTensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, index, top_cell);
  } else {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
  }
}

void DataConvert::MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                             const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  tensor::BaseTensorPtr tensor_ptr = nullptr;
  InputType input_type = InputType::kInput;
  if (v->isa<tensor::BaseTensor>()) {
    tensor_ptr = v->cast<tensor::BaseTensorPtr>();
    if (tensor_ptr->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      op_run_info->op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(tensor_ptr);
    }
  } else if (v->isa<BoolImm>() || v->isa<FloatImm>() || v->isa<Type>() || v->isa<StringImm>() || v->isa<None>()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  } else if (v->isa<IntegerImm>()) {
    if (op_run_info->base_op_run_info.op_name == prim::kPrimCSRReduceSum->name()) {
      int64_t input = v->cast<Int64ImmPtr>()->value();
      op_run_info->op_grad_info->op_prim->set_attr("axis", MakeValue(input));
      return;
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  } else if (v->isa<ValueSequence>()) {
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), index, top_cell);
    return;
  } else if (v->isa<ValueDictionary>()) {
    auto v_dict = v->cast<ValueDictionaryPtr>();
    std::vector<ValuePtr> vec;
    vec.reserve(v_dict->value().size());
    for (const auto &kv : v_dict->value()) {
      (void)vec.emplace_back(kv.second);
    }
    ConvertTupleValueToTensor(op_run_info, std::make_shared<ValueTuple>(vec), index, top_cell);
    return;
  } else if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), top_cell, index);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), top_cell, index);
    return;
  } else if (v->isa<Monad>()) {
    return;
  } else if (v->isa<parse::InterpretedObject>()) {
    MS_EXCEPTION(TypeError) << "Not support for " << v->ToString();
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_types.emplace_back(input_type);
}

void ReplaceReduceAxis(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!common::AnfAlgo::IsReduceOp(op_run_info->base_op_run_info.op_name)) {
    return;
  }
  const auto &inputs = op_run_info->base_op_run_info.expanded_input_values;
  constexpr size_t kReduceOpInputNum = 2;
  if (inputs.size() < kReduceOpInputNum) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size " << inputs.size() << " of Op "
                      << op_run_info->base_op_run_info.op_name;
  }

  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info);
  const auto &op_prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(op_prim);
  if (op_prim->HasAttr(kAttrSkipMode) && GetValue<bool>(op_prim->GetAttr(kAttrSkipMode))) {
    return;
  }

  // 2nd input tensor is {} or nulltpr, means reduce all axis.
  bool reduce_all_axis = false;
  if (inputs[kIndex1]->isa<ValueSequence>()) {
    auto seq_size = inputs[1]->cast<ValueSequencePtr>()->size();
    reduce_all_axis = seq_size == 0;
  } else if (inputs[kIndex1]->isa<None>()) {
    reduce_all_axis = true;
  }
  if (reduce_all_axis) {
    auto size = inputs[0]->cast<tensor::BaseTensorPtr>()->shape().size();
    // For example, input 0 is Tensor(shape=[], value=1), the axis to reduce is 0.
    std::vector<ValuePtr> axis = {std::make_shared<Int64Imm>(0)};
    for (size_t i = 1; i < size; ++i) {
      axis.push_back(std::make_shared<Int64Imm>(static_cast<int64_t>(i)));
    }
    op_run_info->base_op_run_info.expanded_input_values[1] = std::make_shared<ValueTuple>(axis);
  }
}

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);

  (void)op_run_info->base_op_run_info.expanded_input_values.reserve(op_run_info->input_size);
  (void)op_run_info->base_op_run_info.input_types.reserve(op_run_info->input_size);
  // Get input tensors.
  op_run_info->op_grad_info->op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
    // convert const input to attr
    if (RunOpConvertConstInputToAttr(op_run_info, input_object, index)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    MarkInputs(op_run_info, input_object, index, top_cell);
    // -1 indicates input_object is not a dynInput
    if (!op_run_info->base_op_run_info.dyn_input_sizes.empty() && !input_object->isa<ValueSequence>()) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
  }
  op_run_info->op_grad_info->op_prim->EndRecordAddAttr();
  ReplaceReduceAxis(op_run_info);
  AddDynInputsSizesAttr(op_run_info);
}

TopCellInfo *Common::FindPreTopcell(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                                    const std::string &op_info, const ValuePtr &value) {
  const auto &cur_top_cell = grad_executor->top_cell();
  // If the top cell is ir grad, which must be the first step, and pre-top cell cannot be found
  if (cur_top_cell->is_first_step()) {
    // First run top cell, save op output info for replacement
    cur_top_cell->SaveTensorIdWithOpInfo(op_info, value);
    MS_LOG(DEBUG) << "Top cell " << cur_top_cell << " with " << cur_top_cell->already_run_cell_id()
                  << " run firstly, op info " << op_info << ", output id " << Common::GetIdByValue(value);
    // First step or in dynamic process, no need forward output replaces
    op_grad_info->need_do_forward_output_replace = false;
    return nullptr;
  }
  if (cur_top_cell->use_dynamic_shape_process()) {
    MS_LOG(DEBUG) << "Current top cell " << cur_top_cell
                  << "dynamic shape process: " << cur_top_cell->use_dynamic_shape_process();
    return nullptr;
  }
  if (grad_executor->config_no_graph() && !cur_top_cell->is_high_order_top_cell()) {
    return nullptr;
  }
  // Not the first step
  auto pre_top_cell = grad_executor->pre_top_cell();
  MS_EXCEPTION_IF_NULL(pre_top_cell);
  const auto &op_info_with_tensor_object = pre_top_cell->replace_info().op_info_with_tensor_object;
  op_grad_info->used_in_bprop_graph = op_info_with_tensor_object.find(op_info) != op_info_with_tensor_object.end();
  return pre_top_cell;
}

void Common::UpdateGradOpInfo(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                              TopCellInfo *pre_top_cell, bool is_jit_graph) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  const auto &top_cell = grad_executor->top_cell();
  if (!is_jit_graph) {
    grad_executor->dynamic_shape()->CheckNodeDynamic(top_cell, op_grad_info);
  }
  // When config no graph and not high order, we can skip update tensor.
  if (grad_executor->config_no_graph() && !grad_executor->is_high_order_top_cell()) {
    return;
  }
  if (op_grad_info->need_do_forward_output_replace && op_grad_info->used_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(pre_top_cell);
    top_cell->UpdateTopCellForwardTensorInfoInBpropGraph(op_grad_info->op_info, op_grad_info->out_value, pre_top_cell);
  }
}

OperatorType Common::GetOpTypeFromOpdef(const ops::OpDef &op_def) {
  if (op_def.is_view_) {
    return OperatorType::kViewOp;
  }
  if (op_def.returns_[kIndex0].inplace_input_index_ == 0) {
    return OperatorType::kInplaceOp;
  }
  return OperatorType::kDefault;
}

bool GradCommon::IsRealOp(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    return false;
  }
  return kNotRealOP.find(prim->name()) == kNotRealOP.end();
}

bool GradCommon::HasTensorOutput(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    return true;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::any_of(abs_seq.begin(), abs_seq.end(),
                       [](const abstract::AbstractBasePtr &abs) { return HasTensorOutput(abs); });
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto dic_s = abs->cast<abstract::AbstractDictionaryPtr>();
    return std::any_of(dic_s->elements().begin(), dic_s->elements().end(),
                       [](const auto &key_value_pair) { return HasTensorOutput(key_value_pair.second); });
  }
  if (abs->isa<abstract::AbstractCSRTensor>() || abs->isa<abstract::AbstractCOOTensor>()) {
    return true;
  }
  return false;
}

void GradCommon::SetForward(const AnfNodePtrList &node_list) {
  for (const auto &cn : node_list) {
    auto out = Common::CreatOutputTensorValueByAbstract(cn->abstract());
    const auto &c_node = cn->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(c_node);
    c_node->set_forward(Common::CreateValueNodeByValue(out, cn->abstract()), "");
  }
}

void GradCommon::GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                                          AnfNodePtrList *node_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(node_list);
  // Check input used in single op bprop graph. For example,
  // A = a * b;
  // B = A * c;
  // So, A can also replace by its output
  size_t input_num = cnode->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    if (unused_inputs.find(i) == unused_inputs.end() && cnode->input(i + 1)->isa<CNode>()) {
      // Input used by bprop graph, and it is a cnode have produce real output
      const auto &input_c = cnode->input(i + 1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_c);
      if (IsPrimitive(input_c, prim::kPrimMakeTuple)) {
        size_t tuple_input_num = input_c->size() - 1;
        for (size_t j = 0; j < tuple_input_num; ++j) {
          if (auto f_node = common::AnfAlgo::VisitKernel(input_c, j).first;
              f_node->isa<CNode>() && IsRealOp(f_node) && HasTensorOutput(f_node->abstract())) {
            MS_LOG(DEBUG) << "Get used input node " << f_node->DebugString();
            (void)node_list->emplace_back(f_node);
          }
        }
      } else {
        if (auto f_node = common::AnfAlgo::VisitKernel(input_c, 0).first;
            f_node->isa<CNode>() && IsRealOp(f_node) && HasTensorOutput(f_node->abstract())) {
          MS_LOG(DEBUG) << "Get used input node " << f_node->DebugString();
          (void)node_list->emplace_back(f_node);
        }
      }
    }
  }
  // Check output used in single op bprop graph
  if (unused_inputs.find(cnode->size() - 1) == unused_inputs.end() && HasTensorOutput(cnode->abstract())) {
    MS_LOG(DEBUG) << "Get used output node " << cnode->DebugString();
    (void)node_list->emplace_back(cnode);
  }
}
}  // namespace PyNativeAlgo

void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task) {
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    MS_LOG(INFO) << "PyBoost sync run frontend task";
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}
}  // namespace pynative
}  // namespace mindspore
