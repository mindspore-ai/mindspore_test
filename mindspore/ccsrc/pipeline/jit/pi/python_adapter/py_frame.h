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
#ifndef MINDSPORE_PI_JIT_PYTHON_ADAPTER_PY_FRAME_H
#define MINDSPORE_PI_JIT_PYTHON_ADAPTER_PY_FRAME_H

#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "pipeline/jit/pi/python_adapter/py_code.h"

namespace mindspore {
namespace pijit {

/**
 * wrapper Frame object to fast access it's field
 */
class PyFrameWrapper {
 public:
  PyFrameWrapper() : frame_(nullptr) {}
  explicit PyFrameWrapper(EvalFrameObject *f) : frame_(f) {}
  const auto &frame() const { return frame_; }

  // copy the function
  py::object GetFunction() const;

  // copy the free variables
  py::tuple FreeVars() const;

  // copy the locals dict
  py::dict Locals() const;

  // copy the arguments
  py::tuple PackArgs() const;
  py::object Globals() const;
  py::object Builtins() const;

  PyCodeWrapper GetCode() const;
  PyObject **FastLocal() const;

  PyObject *EvalNewCode(PyThreadState *, PyCodeObject *) const;

  template <typename LocalHandler, typename CellHandler, typename FreeHandler>
  void ForEachFastLocal(LocalHandler lh, CellHandler ch = nullptr, FreeHandler fh = nullptr) const {
    auto fast = FastLocal();
    auto code = GetCode();
    for (auto size = code.FastLocalSize(), i = 0; i < size; ++i) {
      auto k = code.FastLocalKind(i);
      if (fast[i] == nullptr) {
        continue;
      }
      if (k == PyCodeWrapper::kCoFastLocal) {
        FastLocalIterFunc(lh)(fast[i], i);
      } else if (k == PyCodeWrapper::kCoFastCell) {
        FastLocalIterFunc(ch)(fast[i], i);
      } else if (k == PyCodeWrapper::kCoFastFree) {
        if (std::is_same<FreeHandler, nullptr_t>::value) {
          break;
        }
        FastLocalIterFunc(fh)(fast[i], i);
      }
    }
  }

 private:
  template <typename Func, typename = std::enable_if_t<!std::is_same<Func, nullptr_t>::value>>
  static constexpr auto FastLocalIterFunc(Func f) {
    return f;
  }
  static constexpr auto FastLocalIterFunc(std::nullptr_t) {
    constexpr auto do_nothing = [](PyObject *, size_t) constexpr {};
    return do_nothing;
  }

  EvalFrameObject *frame_;
};

EvalFrameObject *FrameConvert(PyFrameObject *);

}  // namespace pijit
}  // namespace mindspore

#endif
