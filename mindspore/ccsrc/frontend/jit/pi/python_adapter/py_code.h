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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PY_CODE_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PY_CODE_H

#include <string>
#include <sstream>
#include <map>
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

// the value is -1 if no line or no column
struct CodeLocation;
struct ExceptionTableItem {
  // [begin, end)
  int begin_;
  int end_;
  // handler instruction start index
  int jump_;
  // stack effect
  int stack_;
  // need push last instruction index to stack
  bool lasti_;
};
using ExceptionTable = std::map<int, ExceptionTableItem>;

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
  py::tuple CellVars();
  py::tuple FreeVars();
  py::tuple VarNames();
  py::object Code();
  py::object LineTab() const;
  py::object DeepCopy();

  int Cell2Arg(int cell_var_index);
  int Cell2Arg(const char *cell_var_name);

  std::string ToString() const { return py::str(reinterpret_cast<PyObject *>(ptr())); }
  py::tuple co_consts() const { return py::reinterpret_borrow<py::tuple>(ptr()->co_consts); }
  py::tuple co_names() const { return py::reinterpret_borrow<py::tuple>(ptr()->co_names); }

  enum LocalKind {
    kCoFastLocal,
    kCoFastCell,
    kCoFastFree,
  };
  LocalKind FastLocalKind(int fast_local_index) const;
  int FastLocalIndex(LocalKind kind, int instr_arg) const;
  const char *FastLocalName(int fast_local_index) const;
  py::tuple FastLocalNames() const;
  int FastLocalSize() const;

  int Addr2Line(int byte_offset);
  CodeLocation Addr2Location(int byte_offset);
  ExceptionTable DecodeExceptionTable();

 private:
  PyCodeObject *ptr_;
};

std::string ToString(const PyCodeWrapper &code);

// the value is -1 if no line or no column
struct CodeLocation {
  int start_line_;
  int end_line_;
  int start_column_;
  int end_column_;
};

inline std::ostream &operator<<(std::ostream &s, const CodeLocation &loc) {
  s << "Loc(" << loc.start_line_ << "," << loc.end_line_ << "," << loc.start_column_ << "," << loc.end_column_ << ")";
  return s;
}

inline bool operator==(const CodeLocation &x, const CodeLocation &y) {
  return x.start_line_ == y.start_line_ && x.end_line_ == y.end_line_ && x.start_column_ == y.start_column_ &&
         x.end_column_ == y.end_column_;
}

inline bool operator!=(const CodeLocation &x, const CodeLocation &y) {
  return x.start_line_ != y.start_line_ || x.end_line_ != y.end_line_ || x.start_column_ != y.start_column_ ||
         x.end_column_ != y.end_column_;
}

inline std::ostream &operator<<(std::ostream &s, const ExceptionTableItem &i) {
  s << "[" << i.begin_ << "," << i.end_ << ")->" << i.jump_ << "[" << i.stack_ << "]" << (i.lasti_ ? "lasti" : "");
  return s;
}

}  // namespace pijit
}  // namespace mindspore

#endif
