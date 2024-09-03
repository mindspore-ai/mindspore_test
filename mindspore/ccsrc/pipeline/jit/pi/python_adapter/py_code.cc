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
#include "pipeline/jit/pi/python_adapter/py_code.h"
#if IS_PYTHON_3_11_PLUS
#include "internal/pycore_code.h"
#endif
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {

PyCodeWrapper::PyCodeWrapper(const py::handle &ptr) : ptr_(reinterpret_cast<PyCodeObject *>(ptr.ptr())) {
  if (PyCode_Check(ptr.ptr())) {
    return;
  }
  throw py::type_error("cast to PyCodeObject failed");
}

const char *PyCodeWrapper::Name() const { return PyUnicode_AsUTF8(ptr_->co_name); }
const char *PyCodeWrapper::FileName() const { return PyUnicode_AsUTF8(ptr_->co_filename); }
int PyCodeWrapper::FirstLine() const { return ptr_->co_firstlineno; }
int PyCodeWrapper::LocalSize() const { return ptr_->co_nlocals; }

int PyCodeWrapper::ArgCount(bool *has_var_args, bool *has_kw_var_args) const {
  PyCodeObject *co = this->ptr_;
  const unsigned flags = co->co_flags;
  bool va = flags & CO_VARARGS;
  bool kw_va = flags & CO_VARKEYWORDS;
  has_var_args ? (void)(*has_var_args = va) : (void)0;
  has_kw_var_args ? (void)(*has_kw_var_args = kw_va) : (void)0;
  return co->co_argcount + co->co_kwonlyargcount + va + kw_va;
}

int PyCodeWrapper::PositionOnlyArgCount() const {
#if IS_PYTHON_3_8_PLUS
  PyCodeObject *co = this->ptr_;
  return co->co_posonlyargcount;
#else
  return 0;
#endif
}

int PyCodeWrapper::CellVarsSize() const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return co->co_ncellvars;
#else
  return PyTuple_GET_SIZE(co->co_cellvars);
#endif
}

int PyCodeWrapper::FreeVarsSize() const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return co->co_nfreevars;
#else
  return PyTuple_GET_SIZE(co->co_freevars);
#endif
}

py::tuple PyCodeWrapper::CellVars() {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return py::reinterpret_steal<py::tuple>(PyCode_GetCellvars(co));
#else
  return py::reinterpret_borrow<py::tuple>(co->co_cellvars);
#endif
}

py::tuple PyCodeWrapper::FreeVars() {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return py::reinterpret_steal<py::tuple>(PyCode_GetFreevars(co));
#else
  return py::reinterpret_borrow<py::tuple>(co->co_freevars);
#endif
}

py::tuple PyCodeWrapper::VarNames() {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return py::reinterpret_steal<py::tuple>(PyCode_GetVarnames(co));
#else
  return py::reinterpret_borrow<py::tuple>(co->co_varnames);
#endif
}

py::object PyCodeWrapper::Code() {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return py::reinterpret_steal<py::object>(PyCode_GetCode(co));
#else
  return py::reinterpret_borrow<py::object>(co->co_code);
#endif
}

py::object PyCodeWrapper::LineTab() const {
  PyCodeObject *co = this->ptr_;
  PyObject *line_tab;
#if IS_PYTHON_3_10_PLUS
  line_tab = co->co_linetable;
#else
  line_tab = co->co_lnotab;
#endif
  return py::reinterpret_borrow<py::object>(line_tab);
}

int PyCodeWrapper::FastLocalSize() const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return co->co_nlocalsplus;
#else
  return co->co_nlocals + PyTuple_GET_SIZE(co->co_cellvars) + PyTuple_GET_SIZE(co->co_freevars);
#endif
}

py::tuple PyCodeWrapper::FastLocalNames() const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  return py::reinterpret_borrow<py::tuple>(co->co_localsplusnames);
#else
  PyObject *tmp = PySequence_Concat(co->co_cellvars, co->co_freevars);
  PyObject *res = tmp ? PySequence_Concat(co->co_varnames, tmp) : nullptr;
  Py_XDECREF(tmp);
  return py::reinterpret_steal<py::tuple>(res);
#endif
}

PyCodeWrapper::LocalKind PyCodeWrapper::FastLocalKind(int i) const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  auto kind = _PyLocals_GetKind(co->co_localspluskinds, i);
  if (kind & CO_FAST_LOCAL) {
    return LocalKind::kCoFastLocal;
  } else if (kind & CO_FAST_CELL) {
    return LocalKind::kCoFastCell;
  }
#else
  if (i < co->co_nlocals) {
    return LocalKind::kCoFastLocal;
  } else if (i < co->co_nlocals + PyTuple_GET_SIZE(co->co_cellvars)) {
    return LocalKind::kCoFastCell;
  }
#endif
  // assert i < FastLocalSize
  return LocalKind::kCoFastFree;
}

py::object PyCodeWrapper::DeepCopy() {
#if !IS_PYTHON_3_11_PLUS
  PyCodeObject *co = this->ptr_;
  PyCodeObject *new_code =
    PyCode_New(co->co_argcount, co->co_kwonlyargcount, co->co_nlocals, co->co_stacksize, co->co_flags, Code().ptr(),
               co->co_consts, co->co_names, VarNames().ptr(), FreeVars().ptr(), CellVars().ptr(), co->co_filename,
               co->co_name, co->co_firstlineno, LineTab().ptr());
  if (new_code != nullptr) {
    return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(new_code));
  }
#endif
  throw py::error_already_set();
}

}  // namespace pijit
}  // namespace mindspore
