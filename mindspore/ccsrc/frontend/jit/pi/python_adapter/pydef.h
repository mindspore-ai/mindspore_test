/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PYDEF_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PYDEF_H

#include "frontend/jit/pi/python_adapter/py_macro.h"

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)

extern "C" {
typedef PyObject *(*_PyFrameEvalFunction)(PyFrameObject *, int);
void _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState *state, _PyFrameEvalFunction eval_frame_function);
_PyFrameEvalFunction _PyInterpreterState_GetEvalFrameFunc(PyInterpreterState *state);
}

inline PyObject *_PyEval_EvalFrameDefault(PyThreadState *state, PyFrameObject *f, int exc) {
  return _PyEval_EvalFrameDefault(f, exc);
}

inline PyObject *PyObject_Vectorcall(PyObject *func, PyObject *const *stack, Py_ssize_t nargs, PyObject *kwnames) {
#if PY_MINOR_VERSION == 7
  return _PyObject_FastCallKeywords(func, stack, nargs, kwnames);
#else
  return _PyObject_Vectorcall(func, stack, nargs, kwnames);
#endif
}
#endif

#if IS_PYTHON_3_11_PLUS
#define PY_FRAME_EVAL_FUNCTION_SIGNATURE PyThreadState *ts, _PyInterpreterFrame *f, int exc
#elif IS_PYTHON_3_9_PLUS
#define PY_FRAME_EVAL_FUNCTION_SIGNATURE PyThreadState *ts, PyFrameObject *f, int exc
#else
#define PY_FRAME_EVAL_FUNCTION_SIGNATURE PyFrameObject *f, int exc
#define PY_FRAME_EVAL_FUNCTION_DECLARE_THREAD_STATE() PyThreadState *ts = PyThreadState_Get()
#endif

#endif
