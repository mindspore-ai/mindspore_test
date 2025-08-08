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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "include/securec.h"
#define IS_PYTHON_3_11_PLUS (PY_VERSION_HEX >= 0x030B0000)
#define IS_PYTHON_3_8_PLUS (PY_VERSION_HEX >= 0x03080000)

#if IS_PYTHON_3_11_PLUS
#undef _PyGC_FINALIZED
#endif

#define Py_BUILD_CORE
#if IS_PYTHON_3_8_PLUS
// <stdatomic.h> is unsupported by g++
#include <internal/pycore_pystate.h>
#if IS_PYTHON_3_11_PLUS
#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#endif  // IS_PYTHON_3_11_PLUS
#endif  // IS_PYTHON_3_8_PLUS
#undef Py_BUILD_CORE

#include <frameobject.h>

#define CHECK(expr) assert(expr)
#define CHECK_EQ(expr_a, expr_b) assert((expr_a) == (expr_b))
#define CHECK_GE(expr_a, expr_b) assert((expr_a) >= (expr_b))

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)

_PyFrameEvalFunction _PyInterpreterState_GetEvalFrameFunc(PyInterpreterState *state) { return state->eval_frame; }
void _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState *state, _PyFrameEvalFunction eval_frame_function) {
  state->eval_frame = eval_frame_function;
}

#endif

#if IS_PYTHON_3_11_PLUS

PyCodeObject *EvalFrameGetCode(_PyInterpreterFrame *f) { return f->f_code; }
PyObject *EvalFrameGetGlobals(_PyInterpreterFrame *f) { return f->f_globals; }
PyObject *EvalFrameGetBuiltins(_PyInterpreterFrame *f) { return f->f_builtins; }

PyFunctionObject *EvalFrameGetFunction(_PyInterpreterFrame *f) {
  Py_INCREF(f->f_func);
  return f->f_func;
}

PyObject **EvalFrameGetFastLocals(_PyInterpreterFrame *f) { return _PyFrame_GetLocalsArray(f); }
_PyInterpreterFrame *GetEvalFrame(PyFrameObject *f) { return f->f_frame; }

// here customize, always return a new dict
PyObject *EvalFrameGetLocals(_PyInterpreterFrame *frame) {
  PyObject *locals = PyDict_New();
  if (locals == NULL) {
    return NULL;
  }
  PyCodeObject *co = frame->f_code;
  PyObject **fast = _PyFrame_GetLocalsArray(frame);
  PyObject *closure = frame->f_func->func_closure;
  int offset = co->co_nlocals + co->co_nplaincellvars;

  for (int i = 0; i < co->co_nlocalsplus; i++) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      continue;
    }
    PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
    PyObject *value = fast[i];
    if (frame->stacktop) {
      if (kind & CO_FAST_FREE) {
        value = PyCell_GET(PyTuple_GET_ITEM(closure, i - offset));
      } else if ((kind & CO_FAST_CELL) && value != NULL && PyCell_Check(value)) {
        value = PyCell_GET(value);
      }
    }
    int err = value ? PyDict_SetItem(locals, name, value) : PyDict_DelItem(locals, name);
    if (err) {
      PyErr_Clear();
    }
  }
  return locals;
}

// From https://github.com/python/cpython/blob/3.11/Objects/obmalloc.c#L558
static void *Adapter_PyObject_VirtualAlloc(size_t size) {
  PyObjectArenaAllocator arena;
  PyObject_GetArenaAllocator(&arena);
  return arena.alloc(arena.ctx, size);
}

// From https://github.com/python/cpython/blob/3.11/Objects/obmalloc.c#L564
static void Adapter_PyObject_VirtualFree(void *obj, size_t size) {
  PyObjectArenaAllocator arena;
  PyObject_GetArenaAllocator(&arena);
  arena.free(arena.ctx, obj, size);
}

// From https://github.com/python/cpython/blob/3.11/Python/pystate.c#L2222
static void Adapter_PyThreadState_PopFrame(PyThreadState *tstate, _PyInterpreterFrame *frame) {
  CHECK(tstate->datastack_chunk);
  PyObject **base = (PyObject **)frame;
  if (base == &tstate->datastack_chunk->data[0]) {
    _PyStackChunk *chunk = tstate->datastack_chunk;
    _PyStackChunk *previous = chunk->previous;
    // push_chunk ensures that the root chunk is never popped:
    CHECK(previous);
    tstate->datastack_top = &previous->data[previous->top];
    tstate->datastack_chunk = previous;
    Adapter_PyObject_VirtualFree(chunk, chunk->size);
    tstate->datastack_limit = (PyObject **)(((char *)previous) + previous->size);
  } else {
    CHECK(tstate->datastack_top);
    CHECK(tstate->datastack_top >= base);
    tstate->datastack_top = base;
  }
}

// From https://github.com/python/cpython/blob/3.11/Objects/frameobject.c#L1029
static PyFrameObject *Adapter_PyFrame_New_NoTrack(PyCodeObject *code) {
  CALL_STAT_INC(frame_objects_created);
  int slots = code->co_nlocalsplus + code->co_stacksize;
  PyFrameObject *f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
  if (f == NULL) {
    return NULL;
  }
  f->f_back = NULL;
  f->f_trace = NULL;
  f->f_trace_lines = 1;
  f->f_trace_opcodes = 0;
  f->f_fast_as_locals = 0;
  f->f_lineno = 0;
  return f;
}

// From https://github.com/python/cpython/blob/3.11/Python/frame.c#L27
static PyFrameObject *Adapter_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame) {
  CHECK(frame->frame_obj == NULL);
  PyObject *error_type, *error_value, *error_traceback;
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  PyFrameObject *f = Adapter_PyFrame_New_NoTrack(frame->f_code);
  if (f == NULL) {
    Py_XDECREF(error_type);
    Py_XDECREF(error_value);
    Py_XDECREF(error_traceback);
    return NULL;
  }
  PyErr_Restore(error_type, error_value, error_traceback);
  if (frame->frame_obj) {
    // GH-97002: How did we get into this horrible situation? Most likely,
    // allocating f triggered a GC collection, which ran some code that
    // *also* created the same frame... while we were in the middle of
    // creating it! See test_sneaky_frame_object in test_frame.py for a
    // concrete example.
    //
    // Regardless, just throw f away and use that frame instead, since it's
    // already been exposed to user code. It's actually a bit tricky to do
    // this, since we aren't backed by a real _PyInterpreterFrame anymore.
    // Just pretend that we have an owned, cleared frame so frame_dealloc
    // doesn't make the situation worse:
    f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;
    f->f_frame->owner = FRAME_CLEARED;
    f->f_frame->frame_obj = f;
    Py_DECREF(f);
    return frame->frame_obj;
  }
  CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  CHECK(frame->owner != FRAME_CLEARED);
  f->f_frame = frame;
  frame->frame_obj = f;
  return f;
}

// From https://github.com/python/cpython/blob/3.11/Include/internal/pycore_frame.h#L163
static PyFrameObject *Adapter_PyFrame_GetFrameObject(_PyInterpreterFrame *frame) {
  CHECK(!_PyFrame_IsIncomplete(frame));
  PyFrameObject *res = frame->frame_obj;
  if (res != NULL) {
    return res;
  }
  return Adapter_PyFrame_MakeAndSetFrameObject(frame);
}

// From https://github.com/python/cpython/blob/3.11/Python/frame.c#L27
static void take_ownership(PyFrameObject *f, _PyInterpreterFrame *frame) {
  CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  CHECK(frame->owner != FRAME_CLEARED);
  Py_ssize_t size = ((char *)&frame->localsplus[frame->stacktop]) - (char *)frame;
  if (memcpy_s((_PyInterpreterFrame *)f->_f_frame_data, size, frame, size) != EOK) {
    return;
  }
  frame = (_PyInterpreterFrame *)f->_f_frame_data;
  f->f_frame = frame;
  frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
  if (_PyFrame_IsIncomplete(frame)) {
    // This may be a newly-created generator or coroutine frame. Since it's
    // dead anyways, just pretend that the first RESUME ran:
    PyCodeObject *code = frame->f_code;
    frame->prev_instr = _PyCode_CODE(code) + code->_co_firsttraceable;
  }
  CHECK(!_PyFrame_IsIncomplete(frame));
  CHECK(f->f_back == NULL);
  _PyInterpreterFrame *prev = frame->previous;
  while (prev && _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (prev) {
    /* Link PyFrameObjects.f_back and remove link through _PyInterpreterFrame.previous */
    PyFrameObject *back = Adapter_PyFrame_GetFrameObject(prev);
    if (back == NULL) {
      /* Memory error here. */
      CHECK(PyErr_ExceptionMatches(PyExc_MemoryError));
      /* Nothing we can do about it */
      PyErr_Clear();
    } else {
      f->f_back = (PyFrameObject *)Py_NewRef(back);
    }
    frame->previous = NULL;
  }
  if (!PyObject_GC_IsTracked((PyObject *)f)) {
    PyObject_GC_Track((PyObject *)f);
  }
}

// From https://github.com/python/cpython/blob/3.11/Python/frame.c#L120
static void Adapter_PyFrame_Clear(_PyInterpreterFrame *frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  CHECK(frame->owner != FRAME_OWNED_BY_GENERATOR || _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
  CHECK(_PyThreadState_GET()->cframe->current_frame != frame);
  if (frame->frame_obj) {
    PyFrameObject *f = frame->frame_obj;
    frame->frame_obj = NULL;
    if (Py_REFCNT(f) > 1) {
      take_ownership(f, frame);
      Py_DECREF(f);
      return;
    }
    Py_DECREF(f);
  }
  CHECK_GE(frame->stacktop, 0);
  for (int i = 0; i < frame->stacktop; i++) {
    Py_XDECREF(frame->localsplus[i]);
  }
  Py_XDECREF(frame->frame_obj);
  Py_XDECREF(frame->f_locals);
  Py_DECREF(frame->f_func);
  Py_DECREF(frame->f_code);
}

// From https://github.com/python/cpython/blob/3.11/Python/pystate.c#L728
static _PyStackChunk *allocate_chunk(int size_in_bytes, _PyStackChunk *previous) {
  CHECK_EQ(size_in_bytes % sizeof(PyObject **), 0);
  _PyStackChunk *res = Adapter_PyObject_VirtualAlloc(size_in_bytes);
  if (res == NULL) {
    return NULL;
  }
  res->previous = previous;
  res->size = size_in_bytes;
  res->top = 0;
  return res;
}

#define DATA_STACK_CHUNK_SIZE (16 * 1024)
#define MINIMUM_OVERHEAD 1000
// From https://github.com/python/cpython/blob/3.11/Python/pystate.c#L2182
static PyObject **push_chunk(PyThreadState *tstate, int size) {
  int allocate_size = DATA_STACK_CHUNK_SIZE;
  while (allocate_size < (int)sizeof(PyObject *) * (size + MINIMUM_OVERHEAD)) {
    allocate_size *= 2;
  }
  _PyStackChunk *new = allocate_chunk(allocate_size, tstate->datastack_chunk);
  if (new == NULL) {
    return NULL;
  }
  if (tstate->datastack_chunk) {
    tstate->datastack_chunk->top = tstate->datastack_top - &tstate->datastack_chunk->data[0];
  }
  tstate->datastack_chunk = new;
  tstate->datastack_limit = (PyObject **)(((char *)new) + allocate_size);
  // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
  // _PyThreadState_PopFrame from freeing it later by "skipping" over the
  // first element:
  PyObject **res = &new->data[new->previous == NULL];
  tstate->datastack_top = res + size;
  return res;
}

// From https://github.com/python/cpython/blob/3.11/Python/pystate.c#L2207
static _PyInterpreterFrame *Adapter_PyThreadState_BumpFramePointerSlow(PyThreadState *tstate, size_t size) {
  if (_PyThreadState_HasStackSpace(tstate, size)) {
    _PyInterpreterFrame *res = (_PyInterpreterFrame *)tstate->datastack_top;
    tstate->datastack_top += size;
    return res;
  }
  if (size > INT_MAX / 2) {
    PyErr_NoMemory();
    return NULL;
  }
  return (_PyInterpreterFrame *)push_chunk(tstate, (int)size);
}

_PyInterpreterFrame *EvalFramePushAndInit(PyThreadState *ts, PyFunctionObject *func, PyObject *locals) {
  PyCodeObject *new_co = (PyCodeObject *)func->func_code;
  size_t frame_size = new_co->co_nlocalsplus + new_co->co_stacksize + FRAME_SPECIALS_SIZE;
  _PyInterpreterFrame *new_f = Adapter_PyThreadState_BumpFramePointerSlow(ts, frame_size);
  if (new_f == NULL) {
    return NULL;
  }
  Py_INCREF(func);
  _PyFrame_InitializeSpecials(new_f, func, locals, new_co->co_nlocalsplus);

  PyObject **new_fast = _PyFrame_GetLocalsArray(new_f);
  size_t new_size = new_co->co_nlocalsplus + new_co->co_stacksize;
  if (memset_s(new_fast, sizeof(new_fast[0]) * new_size, 0, sizeof(new_fast[0]) * new_size) != EOK) {
    return NULL;
  }
  return new_f;
}

void EvalFrameClearAndPop(PyThreadState *ts, _PyInterpreterFrame *new_f) {
  CHECK(_PyFrame_GetStackPointer(new_f) == _PyFrame_Stackbase(new_f) ||
        _PyFrame_GetStackPointer(new_f) == new_f->localsplus);
  CHECK((PyObject **)new_f + new_f->f_code->co_nlocalsplus + new_f->f_code->co_stacksize + FRAME_SPECIALS_SIZE ==
        ts->datastack_top);
  CHECK(new_f->frame_obj == NULL || new_f->frame_obj->f_frame == new_f);
  CHECK(new_f->owner == FRAME_OWNED_BY_THREAD);

  Adapter_PyFrame_Clear(new_f);
  Adapter_PyThreadState_PopFrame(ts, new_f);
}

#else

#endif
