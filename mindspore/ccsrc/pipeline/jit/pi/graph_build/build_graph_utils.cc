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
#include "pipeline/jit/pi/graph_build/build_graph_utils.h"

#include <string>
#include <vector>
#include <memory>
#include "utils/flags.h"
#include "utils/ms_context.h"
#include "ir/meta_func_graph.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/pynative/common_utils.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "frontend//ir/primitive_py.h"
#include "frontend/operator/composite/auto_generate/functional_map.h"

namespace mindspore {
namespace pijit {
namespace {
bool IsPrimitiveObject(const py::object &obj) {
  return py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) &&
         parse::data_converter::GetObjType(obj) != parse::RESOLVE_TYPE_CLASS_TYPE;
}

bool IsPrimitiveFunctionalObject(const py::object &obj) {
  return py::isinstance<PrimitiveFunctionAdapter>(obj) || py::isinstance<Functional>(obj);
}

bool IsMsClassObject(const py::object &obj) {
  constexpr auto ms_class_attr = "__ms_class__";
  return py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr));
}

bool IsMetaFuncGraphObject(const py::object &obj) { return py::isinstance<MetaFuncGraph>(obj); }
}  // namespace
void SyncStubTensor(const py::handle &obj) {
  if (!IsStubTensor(obj)) {
    return;
  }
  auto tensor = ConvertStubTensor(obj);
  tensor->data_sync();
}

bool IsSpecialCallableObject(const py::object &obj) {
  static mindspore::HashSet<std::string> func_names{"cast_to_adapter_tensor", "cast_to_ms_tensor"};
  if (!py::hasattr(obj, "__name__")) {
    return false;
  }
  return func_names.find(py::cast<std::string>(obj.attr("__name__"))) != func_names.end();
}

bool IsObjectCallable(const py::object &obj) {
  static constexpr auto check_list = {IsPrimitiveObject, IsPrimitiveFunctionalObject, IsMsClassObject,
                                      IsMetaFuncGraphObject, IsSpecialCallableObject};
  return std::any_of(check_list.begin(), check_list.end(), [&obj](const auto &func) { return func(obj); });
}

bool IsSideEffectPrimitive(const PrimitivePtr &prim) {
  return GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO) || GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
}

py::tuple GetMethodInfo(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
}

bool IsPyCapsuleTensorOverloadMethod(const py::object &obj) {
  if (!IsMsTensorMethod(obj)) {
    return false;
  }
  const auto &info = GetMethodInfo(obj);
  constexpr size_t class_index = 0;
  py::object class_name_obj = info[class_index];
  if (class_name_obj.ptr() == nullptr || py::isinstance<py::none>(class_name_obj)) {
    return false;
  }
  const auto &class_name = class_name_obj.cast<std::string>();
  constexpr auto py_capsule_name = "PyCapsule";
  if (class_name != py_capsule_name) {
    return false;
  }
  constexpr size_t method_index = 1;
  py::object method_name_obj = info[method_index];
  if (method_name_obj.ptr() == nullptr || py::isinstance<py::none>(method_name_obj)) {
    return false;
  }
  const auto &method_name = method_name_obj.cast<std::string>();
  return ops::tensor_method_overload_map.find(method_name) != ops::tensor_method_overload_map.end();
}

bool IsPyCapsuleOverload(const py::object &obj) {
  auto ge_mode = (MsContext::GetInstance()->GetJitLevel() == "O2");
  if (ge_mode) {
    return false;
  }
  return IsPyCapsuleTensorOverloadMethod(obj);
}

bool IsMsTensorMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  return py::cast<bool>(python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_IS_MS_TENSOR_METHOD, obj));
}

bool HasRegisterHook(const py::object &obj) { return HookUtils::HasRegisterHook(obj); }

py::list GetRegisterHookList(const py::object &obj) { return HookUtils::GetRegisterHookList(obj); }

void SaveTensorRegisterHook(const py::object &obj, const AnfNodePtr &node) {
  if (node == nullptr || node->abstract() == nullptr) {
    return;
  }
  auto hook_list = GetRegisterHookList(obj);
  if (hook_list.empty()) {
    return;
  }
  MS_LOG(INFO) << "Save Hook " << py::str(py::object(hook_list)) << " to " << node->DebugString();
  node->abstract()->set_user_data<py::tuple>(kRegisterHookKey, std::make_shared<py::tuple>(py::tuple(hook_list)));
}
}  // namespace pijit
}  // namespace mindspore
