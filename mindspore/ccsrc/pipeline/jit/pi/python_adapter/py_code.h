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
#ifndef MINDSPORE_PI_JIT_PYTHON_ADAPTER_PY_CODE_H
#define MINDSPORE_PI_JIT_PYTHON_ADAPTER_PY_CODE_H

#include <string>
#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

/**
 * wrapper code object to fast access it's field
 */
class PyCodeWrapper {
 public:
  PyCodeWrapper() = default;
  explicit PyCodeWrapper(PyCodeObject *co) : ptr_(co) {}
  explicit PyCodeWrapper(const py::handle &ptr);

  const auto &ptr() const { return ptr_; }

  const char *Name() const;
  const char *FileName() const;
  int FirstLine() const;
  int LocalSize() const;
  int ArgCount(bool *has_var_args = nullptr, bool *has_kw_var_args = nullptr) const;
  int PositionOnlyArgCount() const;
  int CellVarsSize() const;
  int FreeVarsSize() const;
  Py_ssize_t *Cell2Arg();
  py::tuple CellVars();
  py::tuple FreeVars();
  py::tuple VarNames();
  py::object Code();
  py::object LineTab() const;
  py::object DeepCopy();

  int FastLocalSize() const;
  py::tuple FastLocalNames() const;

  std::string ToString() const { return py::str(reinterpret_cast<PyObject *>(ptr())); }
  py::tuple co_consts() const { return py::reinterpret_borrow<py::tuple>(ptr()->co_consts); }
  py::tuple co_names() const { return py::reinterpret_borrow<py::tuple>(ptr()->co_names); }

  enum LocalKind {
    kCoFastLocal,
    kCoFastCell,
    kCoFastFree,
  };
  LocalKind FastLocalKind(int i) const;
  int FastLocalIndex(LocalKind kind, int instr_arg);

 private:
  PyCodeObject *ptr_;
};

std::string ToString(const PyCodeWrapper &code);

}  // namespace pijit
}  // namespace mindspore

#endif
