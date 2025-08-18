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

#include "frontend/jit/pi/python_adapter/py_frame.h"

#if IS_PYTHON_3_11_PLUS

extern "C" {
// borrowed reference
PyCodeObject *EvalFrameGetCode(_PyInterpreterFrame *f);
PyObject *EvalFrameGetGlobals(_PyInterpreterFrame *f);
PyObject *EvalFrameGetBuiltins(_PyInterpreterFrame *f);
_PyInterpreterFrame *GetEvalFrame(PyFrameObject *f);

PyObject **EvalFrameGetFastLocals(_PyInterpreterFrame *f);
_PyInterpreterFrame *EvalFramePushAndInit(PyThreadState *, PyFunctionObject *, PyObject *locals);
void EvalFrameClearAndPop(PyThreadState *, _PyInterpreterFrame *);

// new reference, return a new function object of this frame but no defaults, kwdefaults, annotation, doc
PyFunctionObject *EvalFrameGetFunction(_PyInterpreterFrame *f);
// new reference, return dict
PyObject *EvalFrameGetLocals(_PyInterpreterFrame *f);
}

// new reference, return tuple[CellType]
inline PyObject *EvalFrameGetFreevars(_PyInterpreterFrame *f) {
  auto func = EvalFrameGetFunction(f);
  Py_DECREF(func);  // frame has reference
  return Py_XNewRef(func->func_closure);
}

#else

// borrowed reference
PyCodeObject *EvalFrameGetCode(PyFrameObject *f) { return f->f_code; }
PyObject *EvalFrameGetGlobals(PyFrameObject *f) { return f->f_globals; }
PyObject *EvalFrameGetBuiltins(PyFrameObject *f) { return f->f_builtins; }
PyFrameObject *GetEvalFrame(PyFrameObject *f) { return f; }

PyObject **EvalFrameGetFastLocals(PyFrameObject *f) { return f->f_localsplus; }
PyFrameObject *EvalFramePushAndInit(PyThreadState *ts, PyFunctionObject *func, PyObject *locals) {
  return PyFrame_New(ts, reinterpret_cast<PyCodeObject *>(func->func_code), func->func_globals, locals);
}
void EvalFrameClearAndPop(PyThreadState *, PyFrameObject *f) { Py_DECREF(f); }

// new reference, return tuple[CellType]
PyObject *EvalFrameGetFreevars(PyFrameObject *f) {
  Py_ssize_t size = PyTuple_GET_SIZE(f->f_code->co_freevars);
  PyObject *values = PyTuple_New(size);
  if (values == nullptr) {
    return nullptr;
  }
  Py_ssize_t offset = f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *o = f->f_localsplus[offset + i];
    Py_INCREF(o);
    PyTuple_SET_ITEM(values, i, o);
  }
  return values;
}

// new reference, return a new function object of this frame but no defaults, kwdefaults, annotation, doc
PyFunctionObject *EvalFrameGetFunction(PyFrameObject *f) {
  PyObject *tmp = reinterpret_cast<PyObject *>(f->f_code);
  tmp = PyFunction_New(tmp, f->f_globals);
  PyFunctionObject *new_func = reinterpret_cast<PyFunctionObject *>(tmp);
  if (new_func == nullptr) {
    return nullptr;
  }
  PyObject *closure = EvalFrameGetFreevars(f);
  Py_XSETREF(new_func->func_closure, closure);
  return new_func;
}

// new reference, return dict
PyObject *EvalFrameGetLocals(PyFrameObject *f) {
  PyObject *locals = PyDict_New();
  if (locals == nullptr) {
    return nullptr;
  }
  PyObject **fast = f->f_localsplus;
  PyCodeObject *co = f->f_code;
  PyObject *map[] = {co->co_varnames, co->co_cellvars, co->co_freevars};
  Py_ssize_t size[] = {co->co_nlocals, PyTuple_GET_SIZE(co->co_cellvars), PyTuple_GET_SIZE(co->co_freevars)};
  PyObject *value;
  int err;

  Py_ssize_t j = 0;
  const int map_size = sizeof(map) / sizeof(map[0]);
  for (Py_ssize_t map_index = 0; map_index < map_size; ++map_index) {
    for (Py_ssize_t i = 0; i < size[map_index]; ++i, ++j) {
      value = map_index == 0 ? fast[j] : PyCell_GET(fast[j]);
      if (value == nullptr) {
        err = PyDict_DelItem(locals, PyTuple_GET_ITEM(map[map_index], i));
      } else {
        err = PyDict_SetItem(locals, PyTuple_GET_ITEM(map[map_index], i), value);
      }
      if (err) {
        PyErr_Clear();
      }
    }
  }
  return locals;
}

#endif  // IS_PYTHON_3_11_PLUS

namespace mindspore {
namespace pijit {

// copy a function helper for EvalNewCode
PyFunctionObject *FunctionNew(PyFunctionObject *old_func, PyCodeObject *new_code) {
  PyObject *tmp = reinterpret_cast<PyObject *>(new_code);
  tmp = PyFunction_New(tmp, old_func->func_globals);
  PyFunctionObject *op = reinterpret_cast<PyFunctionObject *>(tmp);
  if (op == nullptr) {
    throw py::error_already_set();
  }
  Py_INCREF(old_func->func_qualname);
  Py_XSETREF(op->func_qualname, old_func->func_qualname);
  Py_XINCREF(old_func->func_defaults);
  Py_XSETREF(op->func_defaults, old_func->func_defaults);
  Py_XINCREF(old_func->func_kwdefaults);
  Py_XSETREF(op->func_kwdefaults, old_func->func_kwdefaults);
  Py_XINCREF(old_func->func_closure);
  Py_XSETREF(op->func_closure, old_func->func_closure);
  Py_XINCREF(old_func->func_annotations);
  Py_XSETREF(op->func_annotations, old_func->func_annotations);
  Py_INCREF(old_func->func_doc);
  Py_XSETREF(op->func_doc, old_func->func_doc);
#if IS_PYTHON_3_11_PLUS
  op->func_version = old_func->func_version;
#endif
  return op;
}

PyObject *PyFrameWrapper::EvalNewCode(PyThreadState *ts, PyCodeObject *new_co) const {
  EvalFrameObject *old_f = this->frame_;
  if (new_co == nullptr) {
    return _PyEval_EvalFrameDefault(ts, old_f, 0);
  }
  PyCodeObject *old_co = EvalFrameGetCode(old_f);
  const unsigned flags = old_co->co_flags;
  constexpr unsigned unsupported_flags = (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR);
  if (flags & unsupported_flags) {
    return nullptr;
  }
  PyFunctionObject *old_func = EvalFrameGetFunction(old_f);
  PyFunctionObject *func = FunctionNew(old_func, new_co);
  Py_DECREF(old_func);
  EvalFrameObject *new_f = EvalFramePushAndInit(ts, func, nullptr);
  Py_DECREF(func);

  PyObject **new_fast = EvalFrameGetFastLocals(new_f);
  PyObject **old_fast = EvalFrameGetFastLocals(old_f);
  PyCodeWrapper new_co_handle(new_co);
  PyCodeWrapper old_co_handle(old_co);
  int new_size = new_co_handle.FastLocalSize();
  int old_size = old_co_handle.FastLocalSize();

  int argc = old_co_handle.ArgCount();
  // assert argc == new_co_handle.ArgCount()

  for (int i = 0; i < argc; ++i) {
    Py_XINCREF(old_fast[i]);
    new_fast[i] = old_fast[i];
  }
  int i = old_size;
  int j = new_size;
  for (; i > old_co->co_nlocals && j > new_co->co_nlocals; --i, --j) {
    PyObject *value = old_fast[i - 1];
    Py_XINCREF(value);
    new_fast[j - 1] = value;
  }
  // assert i == old_co->co_nlocals && j == new_co->co_nlocals

  PyObject *retval = _PyEval_EvalFrameDefault(ts, new_f, 0);
  EvalFrameClearAndPop(ts, new_f);
  return retval;
}

py::tuple PyFrameWrapper::PackArgs() const {
  const int kTwo = 2;
  // CHECK code is execute, args is changed;
  PyObject **fast = EvalFrameGetFastLocals(frame_);
  PyCodeObject *co = EvalFrameGetCode(frame_);
  PyCodeWrapper co_wrapper(co);
  bool has_va;
  bool has_kw_va;
  int argc = co_wrapper.ArgCount(&has_va, &has_kw_va);
  argc = argc - has_va - has_kw_va;
  py::tuple ret_handle(kTwo + 1);
  py::list args_handle(argc);
  PyObject *ret = ret_handle.ptr();
  PyObject *args = args_handle.ptr();

  // set values
  PyObject *value;
  for (int i = 0; i < argc; i++) {
    PyList_SET_ITEM(args, i, Py_XNewRef(fast[i]));
  }
  PyTuple_SET_ITEM(ret, 0, Py_NewRef(args));
  value = has_va ? fast[argc++] : Py_None;
  PyTuple_SET_ITEM(ret, 1, Py_XNewRef(value));
  value = has_kw_va ? fast[argc++] : Py_None;
  PyTuple_SET_ITEM(ret, kTwo, Py_XNewRef(value));
  argc = argc - has_va - has_kw_va;

#if !IS_PYTHON_3_11_PLUS
  Py_ssize_t ncells = PyTuple_GET_SIZE(co->co_cellvars);
  if (co->co_cell2arg != nullptr) {
    for (Py_ssize_t i = 0; i < ncells; ++i) {
      Py_ssize_t argi = co->co_cell2arg[i];
      if (argi != CO_CELL_NOT_AN_ARG) {
        value = PyCell_GET(fast[co->co_nlocals + i]);
        Py_INCREF(value);
        if (argi < argc) {
          PyList_SET_ITEM(args, argi, value);
        } else {
          PyTuple_SET_ITEM(ret, has_va ? (argi == argc ? 1 : kTwo) : kTwo, value);
        }
      }
    }
  }
#endif
  return ret_handle;
}

py::object PyFrameWrapper::GetFunction() const {
  PyFunctionObject *op = EvalFrameGetFunction(frame_);
  PyObject *ref = reinterpret_cast<PyObject *>(op);
  return py::reinterpret_steal<py::object>(ref);
}
py::tuple PyFrameWrapper::FreeVars() const {
  PyObject *ref = EvalFrameGetFreevars(frame_);
  return ref ? py::reinterpret_steal<py::tuple>(ref) : py::tuple();
}
py::dict PyFrameWrapper::Locals() const {
  PyObject *ref = EvalFrameGetLocals(frame_);
  return py::reinterpret_steal<py::dict>(ref);
}
py::object PyFrameWrapper::Globals() const {
  PyObject *ref = EvalFrameGetGlobals(frame_);
  return py::reinterpret_borrow<py::object>(ref);
}
py::object PyFrameWrapper::Builtins() const {
  PyObject *ref = EvalFrameGetBuiltins(frame_);
  return py::reinterpret_borrow<py::object>(ref);
}

PyCodeWrapper PyFrameWrapper::GetCode() const { return PyCodeWrapper(EvalFrameGetCode(frame_)); }

/**
 * Layout of fast locals:
 * Python3.10 and lower:
 * | -- locals(empty if it's cell) -- | -- cells -- | -- frees -- |
 * Cells and frees build at frame producing
 *
 * Python3.11 and higher:
 * | -- locals(and cell arg) -- | -- cells(other) -- | -- frees -- |
 * Cells and fress is empty at the begin of eval frame.
 * Frame is incomplete and no PyFrameObject until executed the first instruction
 */
PyObject *const *PyFrameWrapper::FastLocal() const { return EvalFrameGetFastLocals(frame_); }

}  // namespace pijit
}  // namespace mindspore
