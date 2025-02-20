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

#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include <algorithm>
#include <utility>
#include <set>
#include <queue>
#include "frontend/operator/composite/do_signature.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "mindspore/ops/op_def/arithmetic_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "frontend/operator/composite/unpack_call.h"
#include "include/common/utils/tensor_py.h"
#include "pipeline/jit/pi/graph_build/build_graph_utils.h"

namespace mindspore {
namespace pijit {
namespace {
constexpr auto kPiJitPyObjKey = "pi_jit_py_obj";
constexpr auto kGradFuncPyObject = "grad_func_py_obj";
constexpr auto kGradNetInputs = "grad_net_inputs";
constexpr auto kTensorModule = "mindspore.common";
constexpr auto kAdapterFlag = "adapter_flag";
constexpr auto kInnerOpsModule = "mindspore.ops.operations._inner_ops";
constexpr auto kCandidateIsolatedFlag = "candidate_isolated";

bool Mutable(const py::object &obj, const ValuePtr &value = nullptr) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value != nullptr && value->isa<tensor::MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<tensor::MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

ValuePtr ConvertPyObjToValue(const py::handle &handle) {
  MS_EXCEPTION_IF_NULL(handle.ptr());
  py::object obj = py::reinterpret_borrow<py::object>(handle);
  ValuePtr ret = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    PyRecursionScope rec_check(obj);
    SyncStubTensor(handle);

    if (py::list::check_(obj) || py::tuple::check_(obj) || pijit::IsCellList(obj)) {
      std::vector<ValuePtr> elements;
      for (const auto &i : obj) {
        auto v = ConvertPyObjToValue(i);
        if (v == nullptr) {
          return nullptr;
        }
        elements.push_back(v);
      }
      if (py::list::check_(obj)) {
        return std::make_shared<ValueList>(elements);
      } else {
        return std::make_shared<ValueTuple>(elements);
      }
    }
    if (py::dict::check_(obj)) {
      std::vector<std::pair<ValuePtr, ValuePtr>> elements;
      for (const auto &i : py::cast<py::dict>(obj)) {
        auto k = ConvertPyObjToValue(i.first);
        auto v = ConvertPyObjToValue(i.second);
        if (k == nullptr || v == nullptr) {
          return nullptr;
        }
        elements.push_back(std::make_pair(k, v));
      }
      return std::make_shared<ValueDictionary>(elements);
    }
    if (pijit::IsConvertToInterpretedObject(obj)) {
      return std::make_shared<parse::InterpretedObject>(obj);
    }
    if (parse::ConvertData(obj, &ret)) {
      return ret;
    }
  } catch (const std::exception &e) {
    MS_LOG(INFO) << e.what();
  }
  MS_LOG(INFO) << "Failed to convert python object." << py::str(handle);
  return nullptr;
}

bool HasTensorWithGradData(const ValuePtr &val) {
  if (val == nullptr) {
    return false;
  }
  if (val->isa<ValueSequence>()) {
    const auto &elements = val->cast<ValueSequencePtr>()->value();
    return std::any_of(elements.begin(), elements.end(), [](const auto &e) { return HasTensorWithGradData(e); });
  }

  if (val->isa<ValueDictionary>()) {
    const auto &elements = val->cast<ValueDictionaryPtr>()->value();
    return std::any_of(elements.begin(), elements.end(), [](const auto &e) { return HasTensorWithGradData(e.second); });
  }

  if (!val->isa<tensor::BaseTensor>()) {
    return false;
  }
  auto val_tensor = val->cast<tensor::BaseTensorPtr>();
  auto grad_data = val_tensor->auto_grad_meta_data();
  return grad_data != nullptr && grad_data->input_type() == InputType::kOpOutput;
}

bool NeedBroaden(const py::object &obj, const ValuePtr &value) {
  return TensorArgMutable(obj, value) || Mutable(obj, value) || value->isa<tensor::MetaSparseTensor>();
}

TypeId GetTypeIdFromClassName(const std::string &class_name) {
  static HashMap<std::string, TypeId> class_name_to_type_ids = {
    {"Tensor", kObjectTypeTensorType},   {"list", kObjectTypeList},
    {"tuple", kObjectTypeTuple},         {"int", kNumberTypeInt},
    {"float", kNumberTypeFloat},         {"CellList", kObjectTypeList},
    {"CellDict", kObjectTypeDictionary}, {"dict", kObjectTypeDictionary}};
  auto iter = class_name_to_type_ids.find(class_name);
  if (iter == class_name_to_type_ids.end()) {
    MS_LOG(INFO) << "Failed to convert class name: " << class_name << " to type id.";
    return kTypeUnknown;
  }
  return iter->second;
}

AbstractBasePtr BuildAbstractForInputObject(const py::object &object) {
  if (object.ptr() == nullptr) {
    return nullptr;
  }
  auto value = ConvertPyObjToValue(object);
  if (value == nullptr) {
    return nullptr;
  }
  bool broaden = NeedBroaden(object, value);
  AbstractBasePtr abs = abstract::ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    abs = AbstractBroaden(abs);
  } else if (HasTensorWithGradData(value)) {
    py::object can_be_mutable = python_adapter::CallPyFn("mindspore.common.mutable", "_check_element_type", object);
    // Mutable can only handle scene when all element in python object can be braoden.
    // If input sequence contains element such as None, string, mutable can not add be the input sequence.
    if (!py::bool_(can_be_mutable)) {
      MS_LOG(EXCEPTION) << "Input " << py::str(object) << " contains tensor with gradient but can not mutable.";
    }
    MS_LOG(INFO) << "Input object " << py::str(object) << " has tensor with auto grad data, need broaden";
    abs = AbstractBroaden(abs);
  }
  return abs;
}

bool CheckGraphOutput(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractNamedTuple>()) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &elements = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(), CheckGraphOutput);
  }
  return IsValidOutputAbstractScalar(abs) || IsValidOutputAbstractTensor(abs);
}

bool CheckInvalidCellListDictMethod(const py::object &obj) {
  py::tuple method_info = GetMethodInfo(obj);
  constexpr size_t class_index = 0;
  constexpr size_t method_index = 1;
  py::object class_name_obj = method_info[class_index];
  if (class_name_obj.ptr() == nullptr || py::isinstance<py::none>(class_name_obj)) {
    return false;
  }
  const auto &class_name = class_name_obj.cast<std::string>();
  MS_LOG(INFO) << "class name: " << class_name;
  if (class_name != "CellList" && class_name != "CellDict") {
    return false;
  }
  auto method_name_obj = method_info[method_index];
  if (method_name_obj.ptr() == nullptr || py::isinstance<py::none>(method_name_obj)) {
    return false;
  }
  auto method_name = method_name_obj.cast<std::string>();
  static std::vector<std::string> inplace_method_name = {"clear", "update"};
  if (std::any_of(inplace_method_name.begin(), inplace_method_name.end(),
                  [&method_name](const std::string &name) { return name == method_name; })) {
    MS_LOG(INFO) << "CellDict/CellList inplace function " << method_name << " found";
    return true;
  }
  auto type_id = GetTypeIdFromClassName(class_name);
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  return require.empty();
}

AbstractBasePtr FetchFuncGraphOutputAbstract(const ValuePtr &value) {
  if (value == nullptr || !value->isa<FuncGraph>()) {
    return nullptr;
  }
  auto fg = value->cast<FuncGraphPtr>();
  auto fg_output = fg->output();
  if (fg_output == nullptr) {
    return nullptr;
  }
  return fg_output->abstract();
}

void UpdateParameterFuncGraph(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<Parameter>()) {
    MS_LOG(INFO) << "Input node is not parameter, failed to update graph.";
    return;
  }
  auto param = dyn_cast<Parameter>(node);
  auto origin_fg = param->func_graph();
  auto top_graph = parse::Parser::GetTopFuncGraph();
  if (top_graph == origin_fg) {
    return;
  }
  param->set_func_graph(top_graph);
  MS_LOG(INFO) << "Update parameter function graph from " << origin_fg->ToString() << " to " << top_graph->ToString();
}
}  // namespace

AnfNodePtr FuncGraphBuilder::ConvertParameterTupleToNode(const py::object &input_obj) {
  if (!IsParameterSequence(input_obj)) {
    return nullptr;
  }
  auto tuple_obj = input_obj.cast<py::tuple>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AbstractBasePtr> inputs_abs;
  parse::Resolver resolver(parse::Parser::GetTopFuncGraph());
  for (const auto &obj : tuple_obj) {
    if (!parse::IsParameterObject(py::cast<py::object>(obj))) {
      MS_LOG(INFO) << "Encounter non parameter object in parameter tuple object: " << py::str(obj);
      return nullptr;
    }
    auto cur_node = resolver.ResolveParameterObj(graph_, py::cast<py::object>(obj));
    if (cur_node == nullptr) {
      return nullptr;
    }
    UpdateParameterFuncGraph(cur_node);
    auto cur_abs = cur_node->abstract();
    if (cur_abs == nullptr) {
      return nullptr;
    }
    SaveTensorRegisterHook(py::cast<py::object>(obj), cur_node);
    inputs.push_back(cur_node);
    inputs_abs.push_back(cur_abs);
  }
  auto ret = graph_->NewCNodeInOrder(inputs);
  auto ret_abs = std::make_shared<abstract::AbstractTuple>(inputs_abs);
  ret->set_abstract(ret_abs);
  MS_LOG(INFO) << "Convert parameter tuple to node: " << ret->DebugString()
               << " with abstract: " << ret_abs->ToString();
  return ret;
}

AnfNodePtr FuncGraphBuilder::ConvertObjToNode(const py::object &input_obj) {
  if (input_obj.ptr() == nullptr) {
    MS_LOG(INFO) << "Failed to convert input object to value, python object is null!";
    return nullptr;
  }
  // avoid core dump if converted failed
  ValuePtr val = ConvertPyObjToValue(input_obj);
  if (val == nullptr) {
    MS_LOG(INFO) << "Failed to convert input object to value: " << py::str(input_obj);
    return nullptr;
  }
  if (!parse::ContainsParameter(input_obj)) {
    // Constant value input scene, the object should be converted to value node.
    auto node = NewValueNode(val);
    node->set_abstract(val->ToAbstract());
    return node;
  }
  if (parse::IsParameterObject(input_obj)) {
    // Add the fv parameter and set its abstract.
    parse::Resolver resolver(parse::Parser::GetTopFuncGraph());
    auto ret = resolver.ResolveParameterObj(graph_, input_obj);
    UpdateParameterFuncGraph(ret);
    return ret;
  }
  auto parameter_tuple_object = ConvertParameterTupleToNode(input_obj);
  if (parameter_tuple_object != nullptr) {
    return parameter_tuple_object;
  }
  if (py::isinstance<py::tuple>(input_obj) || py::isinstance<py::list>(input_obj)) {
    return ConvertPyTupleListToNode(input_obj);
  }
  if (py::isinstance<py::dict>(input_obj)) {
    auto dict = input_obj.cast<py::dict>();
    return ConvertPyDictToNode(dict);
  }
  MS_LOG(INFO) << "The Parameter in obj '" << py::str(input_obj) << "' with nested structure is not supported."
               << " Currently only single Parameter, ParameterTuple or Parameters in tuple/list/dict are supported.";
  return nullptr;
}

AnfNodePtr FuncGraphBuilder::ConvertPyTupleListToNode(const py::object &obj) {
  PrimitivePtr prim = py::isinstance<py::tuple>(obj) ? prim::kPrimMakeTuple : prim::kPrimMakeList;
  std::vector<AnfNodePtr> args{NewValueNode(prim)};
  std::vector<AbstractBasePtr> args_abs;

  auto tuple = obj.cast<py::tuple>();
  for (auto &elem : tuple) {
    AnfNodePtr node = ConvertObjToNode(py::cast<py::object>(elem));
    if (node == nullptr || node->abstract() == nullptr) {
      MS_LOG(INFO) << "Failed to convert tuple/list element to node";
      return nullptr;
    }
    args.push_back(node);
    args_abs.push_back(node->abstract());
  }
  auto node = NewCNode(std::move(args), parse::Parser::GetTopFuncGraph());
  node->set_abstract(std::make_shared<abstract::AbstractTuple>(args_abs));
  return node;
}

AnfNodePtr FuncGraphBuilder::ConvertPyDictToNode(const py::dict &dict) {
  std::vector<AnfNodePtr> keys{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> values{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<abstract::AbstractElementPair> kv_abs;
  for (auto &item : dict) {
    AnfNodePtr key = ConvertObjToNode(py::cast<py::object>(item.first));
    AnfNodePtr value = ConvertObjToNode(py::cast<py::object>(item.second));
    if (key == nullptr || value == nullptr || key->abstract() == nullptr || value->abstract() == nullptr) {
      MS_LOG(INFO) << "Failed to convert dict element to node";
      return nullptr;
    }
    keys.push_back(key);
    values.push_back(value);
    (void)kv_abs.emplace_back(std::make_pair(key->abstract(), value->abstract()));
  }
  FuncGraphPtr fg = parse::Parser::GetTopFuncGraph();
  auto node = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), fg->NewCNode(keys), fg->NewCNode(values)});
  node->set_abstract(std::make_shared<abstract::AbstractDictionary>(kv_abs));
  return node;
}

void FuncGraphBuilder::AddLocalVariableNode(const AbstractWrapperPtr &wrapper, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(wrapper);
  MS_EXCEPTION_IF_NULL(node);
  (void)key_to_node_.emplace(wrapper, node);
}

AbstractWrapperPtr FuncGraphBuilder::AddLocalVariable(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, py object is null";
    return nullptr;
  }

  auto node = ConvertObjToNode(obj);
  if (node == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, convert python object to anf node failed";
    return nullptr;
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(node->abstract());

  (void)key_to_node_.emplace(abstract_wrapper, node);
  return abstract_wrapper;
}

AnfNodePtr FuncGraphBuilder::ReadLocalVariable(const AbstractWrapperPtr &abstract_wrapper) {
  auto iter = key_to_node_.find(abstract_wrapper);
  if (iter == key_to_node_.end()) {
    return nullptr;
  }
  return iter->second;
}

AnfNodePtr FuncGraphBuilder::FindNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper) {
  // Search the predecessors of the current builder for the local parameter with BFS.
  if (abstract_wrapper == nullptr || abstract_wrapper->abstract() == nullptr) {
    return nullptr;
  }
  mindspore::HashSet<FuncGraphBuilder *> visited_builders;
  std::queue<FuncGraphBuilder *> builder_queue;
  builder_queue.push(this);
  while (!builder_queue.empty()) {
    const auto cur_builder = builder_queue.front();
    MS_EXCEPTION_IF_NULL(cur_builder);
    builder_queue.pop();
    (void)visited_builders.insert(cur_builder);
    auto node = cur_builder->ReadLocalVariable(abstract_wrapper);
    if (node != nullptr) {
      MS_LOG(INFO) << "Found node: " << node->DebugString()
                   << " for abstract wrapper: " << abstract_wrapper->ToString();
      return node;
    }
    for (const auto &cur_pred_builder : cur_builder->prev_builders()) {
      if (visited_builders.count(cur_pred_builder) == 0) {
        builder_queue.push(cur_pred_builder);
      }
    }
  }
  return nullptr;
}

AnfNodePtr FuncGraphBuilder::FindOrCreateNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper) {
  auto res = FindNodeByWrapper(abstract_wrapper);
  if (res != nullptr) {
    return res;
  }
  if (abstract_wrapper == nullptr || abstract_wrapper->abstract() == nullptr) {
    return nullptr;
  }
  auto abs = abstract_wrapper->abstract();
  MS_LOG(INFO) << "Can't find the AnfNode by wrapper(" << abstract_wrapper.get() << ") abstract is: (" << abs << ") "
               << abs->ToString();
  PrintConstantAbstract(abs);

  // Build ValueNode for constant abstract.
  // Need to handle tuple/list/dict with FuncGraphAbstractClosure scene later.
  auto abstract = abstract_wrapper->abstract();
  if (abstract->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func = abstract->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func->func_graph();
    return NewValueNode(fg);
  }
  auto value = abstract->BuildValue();
  if (!value->ContainsValueAny()) {
    auto ret = NewValueNode(value);
    ret->set_abstract(abstract);
    return ret;
  }
  return nullptr;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphArgInput(const py::object &object) {
  if (object.ptr() == nullptr) {
    MS_LOG(INFO) << "Get top graph arg input failed.";
    return nullptr;
  }
  if (py::isinstance<Cell>(object) || PyFunction_Check(object.ptr()) || PyMethod_Check(object.ptr()) ||
      object.ptr() == Py_None) {
    return nullptr;
  }
  auto abs = BuildAbstractForInputObject(object);
  if (abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for python object: " << std::string(py::str(object)) << "  " << object.ptr();
    return nullptr;
  }
  auto para = AddParameter(graph_);
  para->set_abstract(abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(object));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  origin_top_input_num_ = origin_top_input_num_ + 1;
  MS_LOG(INFO) << "Add top arg input success, python object: " << py::str(object) << ", node: " << para->DebugString()
               << ", abstract: " << abs->ToString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphVargsInputs(const py::object &vargs) {
  if (vargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph vargs is nullptr.";
    return nullptr;
  }
  auto vargs_tuple = vargs.cast<py::tuple>();
  if (vargs_tuple.ptr() == nullptr) {
    MS_LOG(INFO) << "Vargs object should be tuple but got: " << py::str(vargs) << ", add top graph vargs failed.";
    return nullptr;
  }
  auto value = ConvertPyObjToValue(vargs);
  if (value == nullptr || !value->isa<ValueTuple>()) {
    MS_LOG(INFO) << "Convert vargs to value failed, vargs: " << py::str(vargs);
    return nullptr;
  }
  auto value_tuple = value->cast<ValueTuplePtr>();
  const auto &elements = value_tuple->value();
  if (elements.size() != vargs_tuple.size()) {
    MS_LOG(INFO) << "For top graph vargs, converted value element size is " << elements.size()
                 << ", python tuple element size is " << vargs_tuple.size() << ". Size not matched.";
    return nullptr;
  }
  std::vector<AbstractBasePtr> new_elements;
  auto para = AddParameter(graph_);
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_obj = vargs_tuple[i].cast<py::object>();
    auto cur_abs = BuildAbstractForInputObject(cur_obj);
    if (cur_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert args element " << py::str(cur_obj);
      return nullptr;
    }
    new_elements.push_back(cur_abs);
  }
  auto new_vargs_abs = std::make_shared<abstract::AbstractTuple>(new_elements);
  para->set_abstract(new_vargs_abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(vargs));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  MS_LOG(INFO) << "Add top vargs input success, python object: " << py::str(vargs) << ", node: " << para->DebugString()
               << ", abstract: " << new_vargs_abs->ToString();
  origin_top_input_num_ = origin_top_input_num_ + 1;
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddAttributeInput(const py::object &object) {
  if (object.ptr() == nullptr) {
    return nullptr;
  }
  auto value = ConvertPyObjToValue(object);
  if (value == nullptr) {
    return nullptr;
  }
  AbstractBasePtr abs = abstract::ToAbstract(value, nullptr, nullptr);
  if (!abs->isa<abstract::AbstractScalar>() && !abs->isa<abstract::AbstractTensor>()) {
    MS_LOG(ERROR) << "Can not broaden abstract: " << abs->ToString();
    return nullptr;
  }
  abs = AbstractBroaden(abs);
  if (abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for python object: " << std::string(py::str(object)) << "  " << object.ptr();
    return nullptr;
  }
  auto top_graph = parse::Parser::GetTopFuncGraph();
  auto para = AddParameter(top_graph);
  para->set_abstract(abs);
  para->set_is_top_graph_param(true);

  py::object ret_object = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, "convert_to_mutable", object);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(ret_object));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphKwargsInputs(const py::object &kwargs) {
  if (kwargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph kwargs input is nullptr.";
    return nullptr;
  }
  auto kwargs_dict = kwargs.cast<py::dict>();
  if (kwargs_dict.ptr() == nullptr) {
    MS_LOG(INFO) << "Kwargs object should be tuple but got: " << py::str(kwargs) << ", add top graph kwargs failed.";
    return nullptr;
  }
  auto value = ConvertPyObjToValue(kwargs);
  if (value == nullptr || !value->isa<ValueDictionary>()) {
    MS_LOG(INFO) << "Convert kwargs to value failed, kwargs: " << py::str(kwargs);
    return nullptr;
  }
  auto value_dict = value->cast<ValueDictionaryPtr>();
  const auto &elements = value_dict->value();
  if (elements.size() != kwargs_dict.size()) {
    MS_LOG(INFO) << "Kwargs dict size is " << kwargs_dict.size() << " and corresponding value dict size is "
                 << elements.size() << ". Size not matched.";
  }
  auto para = AddParameter(graph_);
  std::vector<abstract::AbstractElementPair> new_key_values;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_key_val = elements[i].first;
    auto cur_val = elements[i].second;
    auto cur_key_obj = ValueToPyData(cur_key_val);
    if (!kwargs_dict.contains(cur_key_obj)) {
      return nullptr;
    }
    auto cur_val_obj = kwargs_dict[cur_key_obj];
    auto cur_value_abs = BuildAbstractForInputObject(cur_val_obj);
    if (cur_value_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert kwargs value element " << py::str(cur_val_obj);
      return nullptr;
    }
    auto cur_key_abs = abstract::ToAbstract(cur_key_val, nullptr, nullptr);
    new_key_values.push_back(abstract::AbstractElementPair{cur_key_abs, cur_value_abs});
  }
  auto new_kwargs_abs = std::make_shared<abstract::AbstractDictionary>(new_key_values);
  para->set_abstract(new_kwargs_abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(kwargs));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  MS_LOG(INFO) << "Add top kwargs input success, python object: " << py::str(kwargs)
               << ", node: " << para->DebugString() << ", abstract: " << new_kwargs_abs->ToString();
  origin_top_input_num_ = origin_top_input_num_ + 1;
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddSubGraphInput(const AbstractWrapperPtr abstract_wrapper) {
  MS_LOG(INFO) << "Try add sub graph parameter for abstract wrapper: " << abstract_wrapper->ToString();
  if (abstract_wrapper == nullptr) {
    MS_LOG(INFO) << "Abstract wrapper for subgraph input is nullptr.";
    return nullptr;
  }
  auto node = FindOrCreateNodeByWrapper(abstract_wrapper);
  if (node == nullptr) {
    MS_LOG(INFO) << "Failed to add input for abstract wrapper: " << abstract_wrapper->ToString();
    return nullptr;
  }
  AbstractBasePtr para_abs = node->abstract();
  if (para_abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for abstract wrapper: " << abstract_wrapper->ToString();
    return nullptr;
  }
  auto para = AddParameter(graph_);
  para->set_abstract(para_abs);
  para->set_is_top_graph_param(false);
  AbstractWrapperPtr ret_abstract_wrapper =
    abstract_wrapper == nullptr ? std::make_shared<AbstractWrapper>(para->abstract()) : abstract_wrapper;
  (void)key_to_node_.emplace(ret_abstract_wrapper, para);
  MS_LOG(INFO) << "Add input success for abstract wrapper: " << abstract_wrapper->ToString()
               << ", result abstract wrapper: " << ret_abstract_wrapper->ToString();
  return ret_abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddNode(const py::object &callable_obj,
                                             const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(callable_obj) << " to value failed.";
    return nullptr;
  }

  const std::string &callable_str = callable_value->ToString();
  const std::string grad_prefix = "MetaFuncGraph-grad";
  if (callable_str.substr(0, grad_prefix.size()) == grad_prefix) {
    MS_LOG(INFO) << "Grad scene callable: " << callable_str;
    return BuildGradNetNode(callable_value, callable_obj, inputs_abstract_wrapper);
  }

  if (IsSpecialCallableObject(callable_obj)) {
    return TryToAddNode(callable_value, inputs_abstract_wrapper);
  }

  return AddNode(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddNodeCallFunctionKw(const ValuePtr &callable_value,
                                                           const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  MS_LOG(INFO) << "Handle CallFunctionKw with callable_value: " << callable_value->ToString();
  auto key_abstract = inputs_abstract_wrapper.back()->abstract();
  if (key_abstract == nullptr || !key_abstract->isa<abstract::AbstractTuple>()) {
    MS_LOG(INFO) << "Key abstract should be tuple but got: " << key_abstract->ToString();
    return nullptr;
  }
  auto key_tuple_abstract = key_abstract->cast<abstract::AbstractTuplePtr>();
  auto key_tuple_value = key_tuple_abstract->BuildValue();
  if (key_tuple_value->ContainsValueAny()) {
    MS_LOG(INFO) << "Key abstract should be constant but got: " << key_abstract->ToString();
    return nullptr;
  }
  size_t dict_len = key_tuple_abstract->size();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs_abstract_wrapper.size() >= dict_len + 1, "kwargs length check error");
  size_t arg_len = inputs_abstract_wrapper.size() - dict_len - 1;

  auto fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> arg_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < arg_len; ++i) {
    auto para = AddParameter(fg);
    (void)arg_inputs.emplace_back(para);
  }
  std::vector<AnfNodePtr> dict_value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < dict_len; ++i) {
    auto para = AddParameter(fg);
    (void)dict_value_inputs.emplace_back(para);
  }
  auto arg_tuple_node = fg->NewCNodeInOrder(arg_inputs);
  auto dict_value_node = fg->NewCNodeInOrder(dict_value_inputs);
  auto dict_key_node = NewValueNode(key_tuple_value);
  auto dict_node_inputs = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), dict_key_node, dict_value_node});
  auto call_node = fg->NewCNodeInOrder(
    {NewValueNode(prim::kPrimDoUnpackCall), NewValueNode(callable_value), arg_tuple_node, dict_node_inputs});
  fg->set_output(call_node);

  AbstractWrapperPtrList new_abstract_wrapper(inputs_abstract_wrapper.begin(), inputs_abstract_wrapper.end() - 1);
  return AddNode(fg, new_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddNodeCallFunctionKw(const py::object &callable_obj,
                                                           const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  MS_LOG(INFO) << "Handle CallFunctionKw with callable_object: " << py::str(callable_obj);
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(INFO) << "Convert to value failed for callable_obj: " << py::str(callable_obj);
    return nullptr;
  }
  return AddNodeCallFunctionKw(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddNodeCallFunctionEx(const ValuePtr &callable_value,
                                                           const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  MS_LOG(INFO) << "Handle CallFunctionKw with callable_value: " << callable_value->ToString();
  auto fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> unpack_call_node_inputs = {NewValueNode(prim::kPrimDoUnpackCall),
                                                     NewValueNode(callable_value)};
  auto first_input_abs = inputs_abstract_wrapper[0]->abstract();
  MS_EXCEPTION_IF_NULL(first_input_abs);
  // First input may be self, need to put into tuple for unpack.
  if (!first_input_abs->isa<abstract::AbstractSequence>() && !first_input_abs->isa<abstract::AbstractDictionary>()) {
    std::vector<AnfNodePtr> self_tuple_node_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    auto para = AddParameter(fg);
    (void)self_tuple_node_inputs.emplace_back(para);
    (void)unpack_call_node_inputs.emplace_back(fg->NewCNodeInOrder(self_tuple_node_inputs));
  } else {
    auto para = AddParameter(fg);
    (void)unpack_call_node_inputs.emplace_back(para);
  }
  for (size_t i = 1; i < inputs_abstract_wrapper.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs_abstract_wrapper[i]);
    auto cur_abstract = inputs_abstract_wrapper[i]->abstract();
    MS_EXCEPTION_IF_NULL(cur_abstract);
    if (!cur_abstract->isa<abstract::AbstractSequence>() && !cur_abstract->isa<abstract::AbstractDictionary>()) {
      MS_LOG(INFO) << "Input abstract should be sequence or dict, but got: " << cur_abstract->ToString();
      return nullptr;
    }
    auto para = AddParameter(fg);
    (void)unpack_call_node_inputs.emplace_back(para);
  }
  auto unpack_call_node = fg->NewCNodeInOrder(unpack_call_node_inputs);
  fg->set_output(unpack_call_node);
  return AddNode(fg, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddNodeCallFunctionEx(const py::object &callable_obj,
                                                           const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  MS_LOG(INFO) << "Handle CallFunctionEx with callable_object: " << py::str(callable_obj);
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(INFO) << "Convert to value failed for callable_obj: " << py::str(callable_obj);
    return nullptr;
  }
  return AddNodeCallFunctionEx(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddAttrPythonObject(const py::object &object) {
  if (object.ptr() == nullptr) {
    MS_LOG(INFO) << "Convert python object with empty object, convert failed.";
    return nullptr;
  }
  // Attribute object is constant or Parameter, do not need to check constant.
  auto node = ConvertObjToNode(object);
  if (node == nullptr || node->abstract() == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(object) << " to anf node failed.";
    return nullptr;
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, node);
  return abstract_wrapper;
}

void FuncGraphBuilder::MarkNodeIsolated(const AnfNodePtr &node, bool force) {
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  auto callable_node = cnode->input(0);
  if (!callable_node->isa<ValueNode>()) {
    return;
  }
  auto callable = callable_node->cast<ValueNodePtr>()->value();
  if (!callable->isa<Primitive>() && !callable->isa<FuncGraph>()) {
    return;
  }
  if (callable->isa<Primitive>()) {
    auto prim = callable->cast<PrimitivePtr>();
    if (force || IsSideEffectPrimitive(prim)) {
      (void)isolated_nodes_.emplace_back(cnode);
      cnode->set_has_side_effect_node(true);
      graph_->set_has_side_effect_node(true);
      MS_LOG(INFO) << "Mark side effect primitive call node isolated, node: " << node->DebugString();
    }
    return;
  }
  auto fg = callable->cast<FuncGraphPtr>();
  if (!force && !fg->has_side_effect_node()) {
    return;
  }
  (void)isolated_nodes_.emplace_back(cnode);
  node->set_user_data<bool>(kCandidateIsolatedFlag, std::make_shared<bool>(true));
  cnode->set_has_side_effect_node(true);
  graph_->set_has_side_effect_node(true);
  MS_LOG(INFO) << "Mark function graph call node isolated, node: " << node->DebugString();
}

void FuncGraphBuilder::EraseCandidateIsolatedNode(const AnfNodePtr &node) {
  if (!(node->has_user_data(kCandidateIsolatedFlag) && *node->user_data<bool>(kCandidateIsolatedFlag))) {
    return;
  }
  if (node->func_graph() != graph_) {
    MS_LOG(INFO) << "Do not erase isolated flag for free variable node: " << node->DebugString();
    return;
  }
  auto iter = std::find(isolated_nodes_.begin(), isolated_nodes_.end(), node);
  if (iter == isolated_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Fail to find node " << node->DebugString() << " from isolated_nodes_";
  }
  isolated_nodes_.erase(iter);
  node->set_user_data<bool>(kCandidateIsolatedFlag, std::make_shared<bool>(false));
  MS_LOG(INFO) << "Erase node " << node->DebugString() << " from isolated_nodes_";
}

bool FuncGraphBuilder::GetInputNodesAndAbstracts(const ValuePtr &callable_value,
                                                 const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                                 AnfNodePtrList *input_node_list, AbstractBasePtrList *input_abs_list) {
  input_node_list->reserve(inputs_abstract_wrapper.size() + 1);
  input_abs_list->reserve(inputs_abstract_wrapper.size());

  (void)input_node_list->emplace_back(NewValueNode(callable_value));
  for (const auto &input_wrapper : inputs_abstract_wrapper) {
    if (input_wrapper == nullptr) {
      MS_LOG(INFO) << "The input python object of " << callable_value->ToString() << ", is NULL";
      return false;
    }
    auto node = FindOrCreateNodeByWrapper(input_wrapper);
    if (node == nullptr) {
      return false;
    }
    EraseCandidateIsolatedNode(node);
    (void)input_node_list->emplace_back(node);
    (void)input_abs_list->emplace_back(node->abstract());
  }
  return true;
}

CNodePtr FuncGraphBuilder::DoPrimitiveInferAndCheck(const PrimitivePtr &primitive,
                                                    const AnfNodePtrList &input_node_list,
                                                    const AbstractBasePtrList &args_abs_list) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    const CNodePtr &new_node = AddPrimitiveCNode(primitive, input_node_list, args_abs_list);
    if (new_node == nullptr) {
      MS_LOG(INFO) << "Failed to add CNode for Primitive: " << primitive->name();
      return nullptr;
    }

    const AbstractBasePtr &abs = BuildNodeAbstract(new_node);
    if (!IsPrimitiveCallable(primitive, abs)) {
      MS_LOG(INFO) << "Check callable failed for Primitive: " << primitive->name();
      return nullptr;
    }
    new_node->set_abstract(abs);
    return new_node;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to infer Primitive: " << primitive->name() << ". The exception:\n" << e.what();
    return nullptr;
  }
}

CNodePtr FuncGraphBuilder::AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                             const AbstractBasePtrList &args_abs_list) {
  auto op_def = mindspore::ops::GetOpDef(primitive->name());

  if (op_def == nullptr) {
    if (primitive->has_signature()) {
      // Follow the implementations in DoSignatureEvaluator
      AnfNodePtrList args_node_list(input_node_list.cbegin() + 1, input_node_list.cend());
      AnfNodePtrList new_node_list =
        prim::GetNewInputsBySignatures(graph_, primitive->ToString(), primitive, args_abs_list, args_node_list);

      new_node_list.insert(new_node_list.begin(), input_node_list[0]);
      return graph_->NewCNodeInOrder(new_node_list);
    }
  } else if (primitive->isa<PrimitivePy>()) {
    // Follow the implementations in PrimitiveArgsToInputsEvaluator and DoTransPrimitiveFunctionEvaluator
    auto arg_signatures = op_def->signatures_;
    primitive->set_signatures(arg_signatures);
    primitive->set_has_signature(!arg_signatures.empty());

    const AnfNodePtrList &init_args = abstract::GetPrimitiveInitArgs(primitive->cast<PrimitivePyPtr>(), op_def);

    AnfNodePtrList call_args(input_node_list.cbegin() + 1, input_node_list.cend());
    AbstractBasePtrList call_abs_list;
    (void)std::transform(call_args.cbegin(), call_args.cend(), std::back_inserter(call_abs_list),
                         [](const AnfNodePtr &node) { return BuildNodeAbstract(node); });
    const AnfNodePtrList &new_call_args =
      prim::GetNewInputsBySignatures(graph_, primitive->name(), primitive, call_abs_list, call_args);

    return abstract::GeneratePrimitiveCNode(primitive, op_def, graph_, init_args, new_call_args,
                                            [](const AnfNodePtr &node) { return BuildNodeAbstract(node); });
  }
  MS_LOG(DEBUG) << "Primitive " << primitive->name() << " no need to process signatures and OpDef";
  return graph_->NewCNodeInOrder(input_node_list);
}

AbstractWrapperPtr FuncGraphBuilder::BuildGradNetNode(const ValuePtr &callable_value, const py::object &callable_obj,
                                                      const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  const std::string grad_prefix = "MetaFuncGraph-grad";
  const std::string fake_node_key_prefix = "FakeNodeKey";
  std::vector<AnfNodePtr> input_node_list;

  (void)input_node_list.emplace_back(NewValueNode(callable_value));
  for (const auto &input_wrapper : inputs_abstract_wrapper) {
    auto node = FindOrCreateNodeByWrapper(input_wrapper);
    if (node == nullptr) {
      // When build grad operation node failed, let forward net run pi jit.
      constexpr size_t forward_net_index = 0;
      auto forward_net_object = AbstractWrapper::FetchPythonObject(inputs_abstract_wrapper[forward_net_index]);
      (void)AbstractWrapper::MarkObjectPiJItShouldCompile(forward_net_object);
      return nullptr;
    }
    (void)input_node_list.emplace_back(node);
  }
  auto fake_node = graph_->NewCNode(input_node_list);

  std::stringstream ss;
  ss << fake_node.get();
  auto output_py_obj = py::str(fake_node_key_prefix + " " + grad_prefix + " " + ss.str());

  auto abs = abstract::ToAbstract(MakeValue(ConvertPyObjToValue(output_py_obj)));
  abs->set_user_data(kGradNetInputs, std::make_shared<AbstractWrapperPtrList>(inputs_abstract_wrapper));
  abs->set_user_data(kGradFuncPyObject, std::make_shared<py::object>(callable_obj));
  fake_node->set_abstract(abs);

  auto abstract_wrapper = std::make_shared<AbstractWrapper>(fake_node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, fake_node);
  MS_LOG(INFO) << "Build GradOperation Net fake node: " << fake_node->DebugString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::BuildGradNode(const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                                   const AbstractWrapperPtrList &inputs) {
  AbstractWrapperPtr ret;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    ret = HandleGrad(key, forward_fg, inputs);
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to build grad node with key: " << key << ". The exception:\n" << e.what();
  }
  return ret;
}

// For GradOperation(net, ...)(forward_inputs), two nodes should be evaluated together as a graph.
// Before:
//   fake_node: GradOperation(net, other_inputs)
// After:
//   fg(other_inputs, forward_inputs)
//     grad_net_node:    DoSignature(GradOperation)(net, other_inputs)
//     grad_result_node: grad_net_node(forward_inputs) or unpack_call(grad_net_node, forward_inputs)
//     return grad_result_node
//   final node for evaluated: fg(other_inputs, forward_inputs)
AbstractWrapperPtr FuncGraphBuilder::HandleGrad(const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                                const AbstractWrapperPtrList &inputs) {
  auto fake_node = ReadLocalVariable(key);
  if (fake_node == nullptr || !fake_node->isa<CNode>()) {
    MS_LOG(INFO) << "Failed to find corresponding fake GradOperation node for key: " << key;
    return nullptr;
  }
  auto fake_node_abstract = fake_node->abstract();
  if (fake_node_abstract == nullptr) {
    MS_LOG(INFO) << "When handling grad, fail to find abstract for fake node: " << fake_node->DebugString();
    return nullptr;
  }
  if (!fake_node_abstract->has_user_data(kGradNetInputs) || !fake_node_abstract->has_user_data(kGradFuncPyObject)) {
    MS_LOG(INFO) << "When handing grad, fail to find corresponding user data for fake node: "
                 << fake_node->DebugString();
    return nullptr;
  }

  auto pre_wrapper = *(fake_node_abstract->user_data<AbstractWrapperPtrList>(kGradNetInputs));
  std::vector<AnfNodePtr> fake_node_inputs;
  for (auto e : pre_wrapper) {
    auto cur_node = FindOrCreateNodeByWrapper(e);
    MS_EXCEPTION_IF_NULL(cur_node);
    fake_node_inputs.push_back(cur_node);
  }

  auto meta_object = *(fake_node_abstract->user_data<py::object>(kGradFuncPyObject));
  auto value = ConvertPyObjToValue(meta_object);
  MS_EXCEPTION_IF_NULL(value);
  auto meta = value->cast<MetaFuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(forward_fg);
  auto origin_forward_fg_output = forward_fg->output();
  auto fake_cnode = fake_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fake_cnode);
  auto meta_node = NewValueNode(std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta));
  std::vector<AnfNodePtr> grad_net_node_inputs{meta_node, NewValueNode(forward_fg)};
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    auto para = AddParameter(fg);
    (void)grad_net_node_inputs.emplace_back(para);
  }
  auto grad_net_node = fg->NewCNodeInOrder(grad_net_node_inputs);
  std::vector<AnfNodePtr> grad_result_node_inputs;
  grad_result_node_inputs.push_back(grad_net_node);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto para = AddParameter(fg);
    (void)grad_result_node_inputs.emplace_back(para);
  }
  auto grad_result_node = fg->NewCNodeInOrder(grad_result_node_inputs);
  fg->set_output(grad_result_node);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR("pijit_grad_fg.ir", fg);
  }
  std::vector<AnfNodePtr> final_node_input = {NewValueNode(fg)};
  std::vector<AbstractBasePtr> final_node_abs;
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    AnfNodePtr cur_input = fake_node_inputs[i];
    MS_EXCEPTION_IF_NULL(cur_input);
    auto cur_input_abs = cur_input->abstract();
    MS_EXCEPTION_IF_NULL(cur_input_abs);
    final_node_input.push_back(cur_input);
    final_node_abs.push_back(cur_input_abs);
  }
  for (auto input_wrapper : inputs) {
    auto node = FindOrCreateNodeByWrapper(input_wrapper);
    MS_EXCEPTION_IF_NULL(node);
    (void)final_node_input.emplace_back(node);
    (void)final_node_abs.emplace_back(node->abstract());
  }
  auto final_node = graph_->NewCNodeInOrder(final_node_input);
  MS_LOG(INFO) << "final_node: " << final_node->DebugString(2);
  for (auto abs : final_node_abs) {
    MS_LOG(INFO) << "final input abstract: " << abs->ToString();
  }
  fg->set_manager(mng_);
  auto analyze_res = pipeline::AbstractAnalyze(fg, final_node_abs);
  MS_EXCEPTION_IF_NULL(analyze_res.eval_result);
  auto final_abs = analyze_res.eval_result->abstract();
  MS_EXCEPTION_IF_NULL(final_abs);
  final_node->set_abstract(final_abs);
  auto cur_forward_fg_output = forward_fg->output();
  if (origin_forward_fg_output != cur_forward_fg_output) {
    // has_aux for GradOperation will change the output of forward fg.
    forward_fg->set_output(origin_forward_fg_output);
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(final_node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, final_node);
  abstract_wrapper->UpdateGradInfo(meta);
  MS_LOG(INFO) << "Build final node " << final_node->DebugString() << " with abstract " << final_abs->ToString();
  return abstract_wrapper;
}

FuncGraphPtr FuncGraphBuilder::BuildCallForwardGraphForGrad(const FuncGraphPtr &fg, const std::vector<size_t> &arg_len,
                                                            bool is_cell) {
  MS_LOG(INFO) << "Build outer fg for vargs scene.";
  auto origin_forward_abs = fg->output()->abstract();
  MS_EXCEPTION_IF_NULL(origin_forward_abs);
  MS_LOG(INFO) << "origin forward abs: " << origin_forward_abs->ToString();

  AnfNodePtrList call_forward_inputs = {NewValueNode(fg)};
  auto outer_fg = std::make_shared<FuncGraph>();
  constexpr auto args_index = 0;
  constexpr auto vargs_index = 1;
  // Eliminate self input for cell when building grad graph.
  size_t input_offset = is_cell ? 1 : 0;
  for (size_t i = 0 + input_offset; i < arg_len[args_index]; ++i) {
    auto para = AddParameter(outer_fg);
    (void)call_forward_inputs.emplace_back(para);
  }
  if (arg_len[vargs_index] != 0) {
    AnfNodePtrList vargs_tuple = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < arg_len[vargs_index]; ++i) {
      auto para = AddParameter(outer_fg);
      (void)vargs_tuple.emplace_back(para);
    }
    auto vargs_node = outer_fg->NewCNodeInOrder(vargs_tuple);
    (void)call_forward_inputs.emplace_back(vargs_node);
  }
  // This is a tmp way to fix empty kwargs.
  if (fg->parameters().size() == call_forward_inputs.size()) {
    (void)call_forward_inputs.emplace_back(NewValueNode(0));
  }
  auto call_forward_node = outer_fg->NewCNodeInOrder(call_forward_inputs);
  call_forward_node->set_abstract(origin_forward_abs);
  outer_fg->set_output(call_forward_node);
  return outer_fg;
}

AbstractWrapperPtr FuncGraphBuilder::TryToAddNode(const ValuePtr &callable_value,
                                                  const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  // Collect the input nodes and input abstracts.
  std::vector<AnfNodePtr> input_node_list;
  std::vector<AbstractBasePtr> input_abs_list;
  if (!GetInputNodesAndAbstracts(callable_value, inputs_abstract_wrapper, &input_node_list, &input_abs_list)) {
    return nullptr;
  }

  AnfNodePtr new_node;
  AbstractBasePtr abs;
  bool is_side_effect = false;
  if (callable_value->isa<Primitive>()) {
    auto prim = callable_value->cast<PrimitivePtr>();
    new_node = DoPrimitiveInferAndCheck(prim, input_node_list, input_abs_list);
    if (new_node != nullptr) {
      abs = new_node->abstract();
    }
    is_side_effect = IsSideEffectPrimitive(prim);
  } else {
    // Do infer and check callable.
    const auto &ret = InferAndCheck(callable_value, input_abs_list);
    abs = ret.first;
    is_side_effect = ret.second;
    if (abs != nullptr) {
      new_node = graph_->NewCNodeInOrder(input_node_list);
    }
  }
  if (new_node == nullptr || abs == nullptr) {
    return nullptr;
  }

  if (!is_side_effect) {
    auto value = abs->BuildValue();
    new_node = value->ContainsValueAny() ? new_node : NewValueNode(value);
  }

  new_node->set_abstract(abs);
  MarkNodeIsolated(new_node, is_side_effect);
  auto ret_abstract_wrapper = std::make_shared<AbstractWrapper>(new_node->abstract());
  (void)key_to_node_.emplace(ret_abstract_wrapper, new_node);
  MS_LOG(INFO) << "Add node: " << new_node->DebugString()
               << " with abstract wrapper: " << ret_abstract_wrapper->ToString();
  return ret_abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddNode(const ValuePtr &callable_value,
                                             const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  if (!callable_value->ToAbstract()->isa<abstract::AbstractFunction>()) {
    MS_LOG(INFO) << "The value " << callable_value->ToString() << " is not callable. The abstract is "
                 << callable_value->ToAbstract()->ToString();
    return nullptr;
  }

  auto ret_abs = FetchFuncGraphOutputAbstract(callable_value);
  if (ret_abs != nullptr) {
    return AddNodeWithAbstract(callable_value, inputs_abstract_wrapper, ret_abs);
  }
  return TryToAddNode(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddMultiNode(const std::string &name,
                                                  const AbstractWrapperPtrList &inputs_abstract_wrapper) {
  const std::string mod_str = "mindspore.ops.composite.multitype_ops";
  py::module mod = py::module::import(mod_str.c_str());
  py::object fn;
  if (py::hasattr(mod, name.c_str())) {
    fn = mod.attr(name.c_str());
  } else {
    const std::string math_ops_mod_str = "mindspore.ops.composite.math_ops";
    py::module math_mod = py::module::import(math_ops_mod_str.c_str());
    if (!py::hasattr(math_mod, name.c_str())) {
      MS_LOG(INFO) << "Fail to find multitype function graph for name " << name;
      return nullptr;
    }
    fn = math_mod.attr(name.c_str());
  }
  return AddNode(fn, inputs_abstract_wrapper);
}

bool FuncGraphBuilder::AddOutput(const AbstractWrapperPtr &abstract_wrapper, bool is_top_graph) {
  if (abstract_wrapper == nullptr) {
    MS_LOG(INFO) << "Fail to add output, abstract wrapper is NULL";
    return false;
  }
  AnfNodePtr node = FindNodeByWrapper(abstract_wrapper);
  if (node == nullptr) {
    MS_LOG(INFO) << "Fail to find correspond anf node for abstract wrapper: " << abstract_wrapper->ToString();
    return false;
  }
  auto abs = node->abstract();
  // Only top graph has restriction on return value type.
  if (is_top_graph && !CheckGraphOutput(abs)) {
    MS_LOG(INFO) << "The output should not be the graph output, abstract: "
                 << (abs == nullptr ? "null" : abs->ToString());
    return false;
  }
  EraseCandidateIsolatedNode(node);
  (void)output_nodes_.emplace_back(node);
  return true;
}

AnfNodePtr FuncGraphBuilder::GenerateOutputNode() {
  if (output_nodes_.size() == 1) {
    return output_nodes_[0];
  }
  output_nodes_.insert(output_nodes_.begin(), NewValueNode(prim::kPrimMakeTuple));
  AbstractBasePtrList abstract_list;
  (void)std::transform(output_nodes_.begin() + 1, output_nodes_.end(), std::back_inserter(abstract_list),
                       [](const AnfNodePtr &node) -> AbstractBasePtr { return node->abstract(); });
  auto output_node = graph_->NewCNodeInOrder(output_nodes_);
  auto fg_output_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  output_node->set_abstract(fg_output_abs);
  return output_node;
}

AnfNodePtr FuncGraphBuilder::AttachIsolatedNode(const AnfNodePtr &node) const {
  if (!graph_->has_side_effect_node()) {
    MS_LOG(DEBUG) << "No side effect node.";
    return node;
  }
  if (isolated_nodes_.empty()) {
    MS_LOG(INFO) << "No isolated node for graph" << graph_->ToString();
    return node;
  }
  AnfNodePtr isolated_node;
  if (isolated_nodes_.size() == 1) {
    isolated_node = isolated_nodes_[0];
  } else {
    AnfNodePtrList isolated_node_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    (void)std::copy(isolated_nodes_.begin(), isolated_nodes_.end(), std::back_inserter(isolated_node_inputs));
    isolated_node = graph_->NewCNodeInOrder(isolated_node_inputs);
  }
  auto stop_gradient_node = graph_->NewCNodeInOrder({NewValueNode(prim::kPrimStopGradient), isolated_node});
  auto ret = graph_->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), node, stop_gradient_node});
  ret->set_abstract(node->abstract());
  return ret;
}

FuncGraphPtr FuncGraphBuilder::graph(bool force) {
  if (has_set_output_ || force) {
    return graph_;
  }
  if (output_nodes_.empty()) {
    MS_LOG(INFO) << "The graph " << graph_->ToString() << " has not been set output.";
    return nullptr;
  }
  bool all_value_node = std::all_of(output_nodes_.begin(), output_nodes_.end(),
                                    [](const AnfNodePtr &node) { return node->isa<ValueNode>(); });
  if (prev_builders().empty() && all_value_node) {
    MS_LOG(INFO) << "All graph output is value node, no need to run graph.";
    return nullptr;
  }
  AnfNodePtr output_node = GenerateOutputNode();
  MS_LOG(INFO) << "Output node before attach isolated node: " << output_node->DebugString();
  output_node = AttachIsolatedNode(output_node);
  MS_LOG(INFO) << "Output node after attach isolated node: " << output_node->DebugString();

  graph_->set_output(output_node);
  has_set_output_ = true;
  return graph_;
}

void FuncGraphBuilder::ClearNodeAbstract() {
  if (!has_set_output_) {
    MS_LOG(INTERNAL_EXCEPTION) << "Graph not generated, can not clear abstract.";
  }
  static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
  auto top_graph = graph();
  if (top_graph == nullptr) {
    return;
  }
  for (const auto &node : mindspore::TopoSort(top_graph->get_return(), SuccDeeperSimple)) {
    MS_EXCEPTION_IF_NULL(node);
    const AbstractBasePtr &prev_inferred = node->abstract();
    if (node->isa<ValueNode>()) {
      PrintConstantAbstract(prev_inferred);
    }
    auto is_func =
      node->isa<mindspore::ValueNode>() && prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>();
    // Keep previous inferred value for parameter and ValueNode if the inferred value is not AbstractFunction.
    if (!node->isa<Parameter>() && !is_func) {
      // Reset tuple/list abstract use flags.
      if (enable_eliminate_unused_element && prev_inferred != nullptr &&
          prev_inferred->isa<abstract::AbstractSequence>()) {
        SetSequenceNodeElementsUseFlags(node, nullptr);
      }
      node->set_abstract(nullptr);
      MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
    }
  }
}

AbstractWrapperPtr FuncGraphBuilder::AddNodeWithAbstract(const ValuePtr &value,
                                                         const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                                         const AbstractBasePtr &abstract) {
  AbstractWrapperPtr ret;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    std::vector<AnfNodePtr> input_node_list;
    input_node_list.reserve(inputs_abstract_wrapper.size() + 1);

    (void)input_node_list.emplace_back(NewValueNode(value));
    for (const auto &input_wrapper : inputs_abstract_wrapper) {
      auto node = FindOrCreateNodeByWrapper(input_wrapper);
      MS_EXCEPTION_IF_NULL(node);
      (void)input_node_list.emplace_back(node);
      EraseCandidateIsolatedNode(node);
    }

    auto new_node = graph_->NewCNodeInOrder(input_node_list);
    new_node->set_abstract(abstract);
    MarkNodeIsolated(new_node, false);

    ret = std::make_shared<AbstractWrapper>(abstract);
    (void)key_to_node_.emplace(ret, new_node);
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to add node with abstract. The exception:\n" << e.what();
  }
  return ret;
}

py::object FuncGraphBuilder::ConvertMethod(const py::object &obj) {
  py::tuple method_info = GetMethodInfo(obj);
  py::object class_name_obj = method_info[0];
  if (py::isinstance<py::none>(class_name_obj)) {
    MS_LOG(INFO) << "Can not get the method info of " << py::str(obj);
    return py::object();
  }
  auto class_name = class_name_obj.cast<std::string>();
  const auto &method_name = method_info[1].cast<std::string>();
  bool is_tensor_method = IsTensorMethod(obj);
  if (class_name == "Tensor" && !is_tensor_method) {
    // object is not method for native tensor.
    return py::object();
  }
  if (is_tensor_method) {
    class_name = "Tensor";
  }

  auto type_id = GetTypeIdFromClassName(class_name);
  MS_LOG(DEBUG) << "type_id: " << type_id << ", method_name: " << method_name;
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(type_id, method_name);
  }

  if (require.empty()) {
    MS_LOG(DEBUG) << "Can not find the method registered.";
    return py::object();
  }

  if (require.is<std::string>()) {
    py::function fn = mindspore::python_adapter::GetPyFn(parse::kStandardMethodModelName, require.cast<std::string>());
    if (py::isinstance<py::none>(fn)) {
      MS_LOG(DEBUG) << "Can not find the method '" << require.cast<std::string>() << "' defined in standard_method.";
      return py::object();
    }
    return fn;
  } else if (require.is<PrimitivePtr>()) {
    auto ops_mod = python_adapter::GetPyModule("mindspore.ops");
    auto primitive_class = python_adapter::GetPyObjAttr(ops_mod, "Primitive");
    return primitive_class(require.cast<PrimitivePtr>()->name());
  }
  MS_LOG(DEBUG) << "The method or attr should be a string or a Primitive, but got " << require.ToString();
  return py::object();
}

py::object FuncGraphBuilder::ConvertFunction(const py::object &obj) {
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_object_map");
  auto callable_obj_ptr = PyDict_GetItem(dict.ptr(), obj.ptr());
  return callable_obj_ptr == nullptr ? py::object() : py::cast<py::object>(callable_obj_ptr);
}

bool FuncGraphBuilder::CanConstantFoldFunc(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object can_constant_fold = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CAN_CONSTANT_FOLD, obj);
  return can_constant_fold.cast<bool>();
}

void FuncGraphBuilder::SetGraphName(const std::string &name) {
  if (name.empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_->debug_info());
  graph_->debug_info()->set_name(name);
}

void FuncGraphBuilder::AddPrevBuilder(const FuncGraphBuilderPtr &builder) { prev_builders_.push_back(builder.get()); }

bool FuncGraphBuilder::ValidateCallableObject(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }
  // Check if object is invalid method for CellList/CellDict, which should not be converted to graph.
  if (CheckInvalidCellListDictMethod(obj)) {
    MS_LOG(INFO) << "The object " << py::str(obj) << " is a invalid CellList/CellDict method, "
                 << "can not convert to graph";
    return false;
  }
  return true;
}
}  // namespace pijit
}  // namespace mindspore
