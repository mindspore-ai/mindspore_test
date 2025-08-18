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
#include "frontend/jit/pi/python_adapter/py_code.h"
#include <string>
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

int PyCodeWrapper::Cell2Arg(int cell_var_index) {
  if (cell_var_index >= CellVarsSize()) {
    return -1;
  }
#if IS_PYTHON_3_11_PLUS
  py::tuple names = CellVars();
  const char *cell_var_name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(names.ptr(), cell_var_index));
  return Cell2Arg(cell_var_name);
#else
  return ptr_->co_cell2arg != nullptr && ptr_->co_cell2arg[cell_var_index] != CO_CELL_NOT_AN_ARG
           ? ptr_->co_cell2arg[cell_var_index]
           : -1;
#endif
}

int PyCodeWrapper::Cell2Arg(const char *cell_var_name) {
#if IS_PYTHON_3_11_PLUS
  py::tuple names = VarNames();
  for (int i = 0, size = names.size(); i < size; ++i) {
    const char *name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(names.ptr(), i));
    if (strcmp(name, cell_var_name) == 0) {
      return i;
    }
  }
#else
  py::tuple names = CellVars();
  for (int i = 0, size = names.size(); i < size; ++i) {
    const char *name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(names.ptr(), i));
    if (strcmp(name, cell_var_name) == 0) {
      return Cell2Arg(i);
    }
  }
#endif
  return -1;
}

PyCodeWrapper::LocalKind PyCodeWrapper::FastLocalKind(int i) const {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  auto kind = _PyLocals_GetKind(co->co_localspluskinds, i);
  if (kind & CO_FAST_FREE) {
    return LocalKind::kCoFastFree;
  }
  if (kind & CO_FAST_CELL) {
    return LocalKind::kCoFastCell;
  }
  if (kind & CO_FAST_LOCAL) {
    return LocalKind::kCoFastLocal;
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

int PyCodeWrapper::FastLocalIndex(PyCodeWrapper::LocalKind kind, int instr_arg) const {
  if (kind == LocalKind::kCoFastLocal) {
    return instr_arg;
  }
  if (kind == LocalKind::kCoFastCell || kind == LocalKind::kCoFastFree) {
#if IS_PYTHON_3_11_PLUS
    return instr_arg;
#else
    return ptr_->co_nlocals + instr_arg;
#endif
  }
  return -1;
}

const char *PyCodeWrapper::FastLocalName(int fast_local_index) const {
  PyCodeObject *co = this->ptr_;

#if IS_PYTHON_3_11_PLUS
  return PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_localsplusnames, fast_local_index));
#else
  if (fast_local_index < co->co_nlocals) {
    return PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_varnames, fast_local_index));
  } else if (fast_local_index < FastLocalSize()) {
    int size = PyTuple_GET_SIZE(co->co_cellvars);
    int index = fast_local_index - co->co_nlocals;
    return PyUnicode_AsUTF8(index < size ? PyTuple_GET_ITEM(co->co_cellvars, index)
                                         : PyTuple_GET_ITEM(co->co_freevars, index - size));
  }
#endif
  return nullptr;
}

py::object PyCodeWrapper::DeepCopy() {
  PyCodeObject *co = this->ptr_;
#if IS_PYTHON_3_11_PLUS
  PyCodeObject *new_code =
    PyCode_New(co->co_argcount, co->co_kwonlyargcount, co->co_nlocals, co->co_stacksize, co->co_flags, Code().ptr(),
               co->co_consts, co->co_names, VarNames().ptr(), FreeVars().ptr(), CellVars().ptr(), co->co_filename,
               co->co_name, co->co_qualname, co->co_firstlineno, LineTab().ptr(), co->co_exceptiontable);
#else
  PyCodeObject *new_code =
    PyCode_New(co->co_argcount, co->co_kwonlyargcount, co->co_nlocals, co->co_stacksize, co->co_flags, Code().ptr(),
               co->co_consts, co->co_names, VarNames().ptr(), FreeVars().ptr(), CellVars().ptr(), co->co_filename,
               co->co_name, co->co_firstlineno, LineTab().ptr());
#endif
  if (new_code != nullptr) {
    return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(new_code));
  }

  throw py::error_already_set();
}

int PyCodeWrapper::Addr2Line(int byte_offset) { return PyCode_Addr2Line(ptr_, byte_offset); }
CodeLocation PyCodeWrapper::Addr2Location(int byte_offset) {
#if IS_PYTHON_3_11_PLUS
  CodeLocation loc;
  PyCode_Addr2Location(ptr_, byte_offset, &loc.start_line_, &loc.start_column_, &loc.end_line_, &loc.end_column_);
  return loc;
#else
  int line = Addr2Line(byte_offset);
  return {line, line, -1, -1};
#endif
}

ExceptionTable PyCodeWrapper::DecodeExceptionTable() {
  ExceptionTable exc_table;
  py::object bytes = py::getattr(reinterpret_cast<PyObject *>(ptr()), "co_exceptiontable", nullptr);
  if (bytes.ptr() == nullptr) {
    return exc_table;
  }
  const uint8_t *begin = reinterpret_cast<const uint8_t *>(PyBytes_AsString(bytes.ptr()));
  const uint8_t *end = begin + PyBytes_GET_SIZE(bytes.ptr());
  auto iter = begin;
  const auto read_variant = [&iter, &end]() {
    const int kValidBits = 6;
    const int kValueMask = (1 << kValidBits) - 1;
    int val = iter[0] & kValueMask;
    while (iter[0] & (1 << kValidBits)) {
      iter++;
      val = (val << kValidBits) | (iter[0] & kValueMask);
    }
    ++iter;
    return val;
  };
  while (iter < end) {
    int start = read_variant();
    int len = read_variant();
    int jump = read_variant();
    int pack = read_variant();
    exc_table[start] = {start, start + len, jump, pack >> 1, (pack & 1) ? true : false};
  }
  return exc_table;
}

std::string ToString(const PyCodeWrapper &code) {
  return std::string(py::str(reinterpret_cast<PyObject *>(code.ptr())));
}

}  // namespace pijit
}  // namespace mindspore
