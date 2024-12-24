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
#include <vector>
#include "third_party/securec/include/securec.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// Helper function to process and assign the docstring
void assign_docstring(const char **target_doc, py::str doc_str) {
  const char *doc_cstr = PyUnicode_AsUTF8(doc_str.ptr());
  if (doc_cstr == nullptr) {
    throw py::error_already_set();
  }

  size_t doc_len = strlen(doc_cstr);
  char *doc_copy = reinterpret_cast<char *>(malloc(doc_len + 1));
  if (doc_copy == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for docstring");
    throw py::error_already_set();
  }

  int result = snprintf_s(doc_copy, doc_len + 1, doc_len, "%s", doc_cstr);
  if (result < 0 || static_cast<size_t>(result) >= doc_len + 1) {
    free(doc_copy);
    PyErr_SetString(PyExc_RuntimeError, "Failed to copy docstring safely");
    throw py::error_already_set();
  }

  if (*target_doc != nullptr) {
    free(const_cast<char *>(*target_doc));
  }

  *target_doc = doc_copy;
}

py::object add_docstr(py::object obj, py::str doc_str) {
  // Check if the object is an instance method (PyInstanceMethod_Type)
  if (Py_TYPE(obj.ptr()) == &PyInstanceMethod_Type) {
    PyInstanceMethodObject *meth = reinterpret_cast<PyInstanceMethodObject *>(obj.ptr());
    // The underlying function object
    PyObject *func_obj = meth->func;
    if (PyCFunction_Check(func_obj)) {
      PyCFunctionObject *func = reinterpret_cast<PyCFunctionObject *>(func_obj);
      assign_docstring(&func->m_ml->ml_doc, doc_str);
    } else {
      PyErr_Format(PyExc_TypeError, "Cannot add docstring to non-CFunction instance method");
      throw py::error_already_set();
    }
  } else {
    PyErr_Format(PyExc_TypeError, "Invalid type '%s' to add docstring", Py_TYPE(obj.ptr())->tp_name);
    throw py::error_already_set();
  }

  Py_INCREF(obj.ptr());
  return obj;
}

void RegTensorDoc(py::module *m) { (void)m->def("_add_docstr", add_docstr); }
}  // namespace mindspore
