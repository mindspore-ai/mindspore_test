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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PY_MACRO_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_PYTHON_ADAPTER_PY_MACRO_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
#include <frameobject.h>
#endif

#define IS_PYTHON_3_7_PLUS (PY_VERSION_HEX >= 0x03070000)
#define IS_PYTHON_3_8_PLUS (PY_VERSION_HEX >= 0x03080000)
#define IS_PYTHON_3_9_PLUS (PY_VERSION_HEX >= 0x03090000)
#define IS_PYTHON_3_10_PLUS (PY_VERSION_HEX >= 0x030A0000)
#define IS_PYTHON_3_11_PLUS (PY_VERSION_HEX >= 0x030B0000)
#define IS_PYTHON_3_12_PLUS (PY_VERSION_HEX >= 0x030C0000)
#define IS_PYTHON_3_13_PLUS (PY_VERSION_HEX >= 0x030D0000)

#ifndef _PyCode_NBYTES
#define _PyCode_NBYTES(co) (PyBytes_GET_SIZE((co)->co_code))
#endif  // _PyCode_NBYTES

#ifndef _PyCode_CODE
#define _PyCode_CODE(co) (reinterpret_cast<_Py_CODEUNIT *>(PyBytes_AS_STRING((co)->co_code)))
#endif  // _PyCode_CODE

#ifndef Py_NewRef
static inline PyObject *_Py_NewRef(PyObject *op) {
  Py_INCREF(op);
  return op;
}
static inline PyObject *_Py_XNewRef(PyObject *op) {
  Py_XINCREF(op);
  return op;
}
#define Py_NewRef(op) (_Py_NewRef(op))
#define Py_XNewRef(op) (_Py_XNewRef(op))
#endif  // Py_NewRef

#ifndef Py_IS_TYPE
#define Py_IS_TYPE(ob, type) (Py_TYPE(ob) == type)
#endif  // Py_IS_TYPE

#ifndef _Py_MAKECODEUNIT
#ifdef WORDS_BIGENDIAN
#define _Py_MAKECODEUNIT(opcode, oparg) (((opcode) << 8) | (oparg))
#else
#define _Py_MAKECODEUNIT(opcode, oparg) ((opcode) | ((oparg) << 8))
#endif
#endif  // _Py_MAKECODEUNIT

#ifndef CO_CELL_NOT_AN_ARG
#define CO_CELL_NOT_AN_ARG -1
#endif  // CO_CELL_NOT_AN_ARG

#endif
