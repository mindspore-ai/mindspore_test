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
#include "include/common/pybind_api/api_register.h"
#include "pipeline/jit/pi/auto_grad/function_node.h"
#include "pipeline/jit/pi/external.h"
namespace mindspore {
namespace pijit {
namespace py = pybind11;
using FunctionNode = mindspore::pijit::grad::FunctionNode;

// Interface with python
void RegPIJitInterface(py::module *m) {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 12)
  // pijit cannot support python>=3.12 for now, but will be adapted very soon.
  (void)m->def("jit_mode_pi_enable", []() {
    MS_LOG(ERROR) << "not support python3.11 bytecode yet.";
    return py::bool_(false);
  });
  (void)m->def("jit_mode_pi_disable", []() { return py::bool_(false); });
  (void)m->def("jit_mode_pi_compile",
               [](const py::object &, const py::object &, const py::object &) { return py::bool_(false); });
  (void)m->def(
    "update_pijit_default_config", [](py::args, py::kwargs) { return py::none(); }, "update pijit default config");
  (void)m->def(
    "get_code_extra", [](py::args, py::kwargs) { return py::none(); },
    "get copy of code extra which is the pijit compile result");
  (void)m->def(
    "function_id", [](py::args, py::kwargs) { return py::int_(0); },
    "Get cpp function pointer, or python function pointer, or object pointer");
#else
  // PIJit interface
  (void)m->def("jit_mode_pi_enable", &mindspore::pi_jit_enable, "enable jit from python byte code");
  (void)m->def("jit_mode_pi_disable", &mindspore::pi_jit_disable, "disable jit from python byte code");
  (void)m->def("jit_mode_pi_compile", &mindspore::pi_jit_should_compile, "add function to compile");
  (void)m->def("update_pijit_default_config", &mindspore::update_pijit_default_config, "update pijit default config");
  (void)m->def("get_code_extra", &mindspore::get_code_extra,
               "get copy of code extra which is the pijit compile result");

  (void)m->def("function_id", &mindspore::FunctionId,
               "Get cpp function pointer, or python function pointer, or object pointer");

  (void)py::class_<FunctionNode, mindspore::pijit::grad::FunctionNodePtr>(*m, "FunctionNode_")
    .def_static("record_primitive", &FunctionNode::RecordPrimitive, py::arg("prim"), py::arg("out"), py::arg("inputs"),
                "Record the executed primitive during forward execution.")
    .def("apply", &FunctionNode::Apply, py::arg("grad"), "Calculate the gradient of the function node.");
#endif
}
}  // namespace pijit
}  // namespace mindspore
