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
  py::tuple CellVars();
  py::tuple FreeVars();
  py::tuple VarNames();
  py::object Code();
  py::object LineTab() const;
  py::object DeepCopy();

  int FastLocalSize() const;
  py::tuple FastLocalNames() const;

  enum LocalKind {
    kCoFastLocal,
    kCoFastCell,
    kCoFastFree,
  };
  LocalKind FastLocalKind(int i) const;

 private:
  PyCodeObject *ptr_;
};

}  // namespace pijit
}  // namespace mindspore

#endif
