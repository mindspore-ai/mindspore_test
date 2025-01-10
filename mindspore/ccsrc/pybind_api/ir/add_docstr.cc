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

#include <string>
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
py::object add_docstr(py::object obj, py::str doc_str) {
  // Check if the object is an instance method (PyInstanceMethod_Type)
  if (Py_TYPE(obj.ptr()) == &PyInstanceMethod_Type) {
    PyInstanceMethodObject *meth = reinterpret_cast<PyInstanceMethodObject *>(obj.ptr());
    // The underlying function object
    PyObject *func_obj = meth->func;
    if (PyCFunction_Check(func_obj)) {
      PyCFunctionObject *func = reinterpret_cast<PyCFunctionObject *>(func_obj);
      // Directly convert doc_str to std::string
      const std::string &new_doc = doc_str.cast<std::string>();

      // Allocate a new C-style string for ml_doc (as it expects a char*)
      char *doc_cstr = strdup(new_doc.c_str());
      if (!doc_cstr) {
        throw std::runtime_error("Memory allocation failed for docstring");
      }

      // Free the existing ml_doc if necessary and assign the new docstring
      if (func->m_ml->ml_doc) {
        free(const_cast<char *>(func->m_ml->ml_doc));
      }
      func->m_ml->ml_doc = doc_cstr;
    } else {
      throw std::invalid_argument("Cannot add docstring to non-CFunction instance method");
    }
  } else {
    throw std::invalid_argument(std::string("Invalid type '") + Py_TYPE(obj.ptr())->tp_name + "' to add docstring");
  }

  Py_INCREF(obj.ptr());
  return obj;
}

void RegTensorDoc(py::module *m) { (void)m->def("_add_docstr", add_docstr); }
}  // namespace mindspore
