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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PYTHON_OBJ_POINTER_H_
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PYTHON_OBJ_POINTER_H_

#include "pybind11/pybind11.h"

namespace mindspore {
namespace profiler {

template <class T>
class PythonObjPointer {
 public:
  PythonObjPointer() : ptr(nullptr) {}
  explicit PythonObjPointer(T *ptr) noexcept : ptr(ptr) {}
  PythonObjPointer(PythonObjPointer &&p) noexcept : ptr(std::exchange(p.ptr, nullptr)) {}
  ~PythonObjPointer() { FreePtr(); }

  T *get() { return ptr; }
  const T *get() const { return ptr; }
  operator T *() { return ptr; }
  T *operator->() { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }
  T *release() {
    T *tmp = ptr;
    ptr = nullptr;
    return tmp;
  }
  PythonObjPointer &operator=(PythonObjPointer &&p) noexcept {
    FreePtr();
    ptr = p.ptr;
    p.ptr = nullptr;
    return *this;
  }
  PythonObjPointer &operator=(T *new_ptr) noexcept {
    FreePtr();
    ptr = new_ptr;
    return *this;
  }

 private:
  void FreePtr();
  T *ptr = nullptr;
};

template <>
void PythonObjPointer<PyObject>::FreePtr() {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 11)
  if (ptr && Py_IsInitialized()) {
    Py_DECREF(ptr);
  }
#endif
}
template class PythonObjPointer<PyObject>;

template <>
void PythonObjPointer<PyCodeObject>::FreePtr() {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 11)
  if (ptr && Py_IsInitialized()) {
    Py_DECREF(ptr);
  }
#endif
}
template class PythonObjPointer<PyCodeObject>;

template <>
void PythonObjPointer<PyFrameObject>::FreePtr() {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 11)
  if (ptr && Py_IsInitialized()) {
    Py_DECREF(ptr);
  }
#endif
}
template class PythonObjPointer<PyFrameObject>;

using PythonObjPtr = PythonObjPointer<PyObject>;
using PythonCodeObjPtr = PythonObjPointer<PyCodeObject>;
using PythonFrameObjPtr = PythonObjPointer<PyFrameObject>;
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PYTHON_OBJ_POINTER_H_
