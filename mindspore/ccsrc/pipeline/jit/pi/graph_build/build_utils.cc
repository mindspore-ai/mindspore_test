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
#include "pipeline/jit/pi/graph_build/build_utils.h"

#include <string>
#include <vector>
#include <memory>
#include "utils/flags.h"
#include "ir/meta_func_graph.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pybind_api/ir/primitive_py.h"
#include "pipeline/pynative/grad/variable.h"

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
  static std::vector<std::function<bool(const py::object &)>> check_list{
    IsPrimitiveObject, IsPrimitiveFunctionalObject, IsMsClassObject, IsMetaFuncGraphObject, IsSpecialCallableObject};
  return std::any_of(check_list.cbegin(), check_list.cend(), [&obj](const auto &func) { return func(obj); });
}

bool IsSideEffectPrimitive(const PrimitivePtr &prim) {
  return GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO) || GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
}

bool HasRegisterHook(const py::object &obj) {
  if (!py::isinstance<tensor::BaseTensor>(obj)) {
    return false;
  }
  auto tensor = py::cast<tensor::BaseTensorPtr>(obj);
  const auto &grad_meta_data = pynative::autograd::impl::get_autograd_meta_impl(tensor);
  if (grad_meta_data == nullptr || !grad_meta_data->is_register_hook()) {
    return false;
  }
  return !grad_meta_data->backward_hooks().empty();
}

py::list GetRegisterHookList(const py::object &obj) {
  if (!HasRegisterHook(obj)) {
    return py::list();
  }
  py::list hook_fn_list;
  auto tensor = py::cast<tensor::BaseTensorPtr>(obj);
  const auto &grad_meta_data = pynative::autograd::impl::get_autograd_meta_impl(tensor);
  const auto &backward_hooks = grad_meta_data->backward_hooks();
  for (const auto &[id, hook] : backward_hooks) {
    auto fn = hook->hook_;
    if (py::isinstance<py::none>(fn)) {
      MS_LOG(DEBUG) << "Hook of Tensor[" << id << "] is None.";
      continue;
    }
    hook_fn_list.append(fn);
  }
  return hook_fn_list;
}

void SaveTensorRegisterHook(const py::object &obj, const AnfNodePtr &node) {
  if (node->abstract() == nullptr) {
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
