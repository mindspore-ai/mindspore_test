/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <utility>
#include <vector>
#include "pybind_api/ir/tensor/storage/storage_py.h"
#include "include/utils/exception.h"
#include "include/utils/pybind_api/api_register.h"

namespace mindspore {
static Py_ssize_t StoragePy_Length(StoragePy *self) {
  HANDLE_MS_EXCEPTION
  return static_cast<Py_ssize_t>(StoragePy_Unpack(self).NBytes());
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *StoragePy_Getitem(StoragePy *self, PyObject *index) {
  HANDLE_MS_EXCEPTION
  MS_LOG(EXCEPTION) << "The function __getitem__ is not implemented for Storage!";
  HANDLE_MS_EXCEPTION_END
}

static int StoragePy_Setitem(StoragePy *self, PyObject *index, PyObject *value) {
  HANDLE_MS_EXCEPTION
  MS_LOG(EXCEPTION) << "The function __setitem__ is not implemented for Storage!";
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static void StoragePy_PyDealloc(PyObject *obj) {
  const auto &storage = StoragePy_Unpack(obj);
  storage.~Storage();
  Py_TYPE(obj)->tp_free(obj);
}

static PyMappingMethods StoragePy_mappingmethods = {(lenfunc)StoragePy_Length, (binaryfunc)StoragePy_Getitem,
                                                    (objobjargproc)StoragePy_Setitem};

PyTypeObject StoragePyType = {
  PyVarObject_HEAD_INIT(NULL, 0) "StoragePy", /* tp_name */
  sizeof(StoragePy),                          /* tp_basicsize */
  0,                                          /* tp_itemsize */
  StoragePy_PyDealloc,                        /* tp_dealloc */
  0,                                          /* tp_vectorcall_offset */
  nullptr,                                    /* tp_getattr */
  nullptr,                                    /* tp_setattr */
  nullptr,                                    /* tp_reserved */
  nullptr,                                    /* tp_repr */
  nullptr,                                    /* tp_as_number */
  nullptr,                                    /* tp_as_sequence */
  &StoragePy_mappingmethods,                  /* tp_as_mapping */
  nullptr,                                    /* tp_hash  */
  nullptr,                                    /* tp_call */
  nullptr,                                    /* tp_str */
  nullptr,                                    /* tp_getattro */
  nullptr,                                    /* tp_setattro */
  nullptr,                                    /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
  nullptr,                                    /* tp_doc */
  nullptr,                                    /* tp_traverse */
  nullptr,                                    /* tp_clear */
  nullptr,                                    /* tp_richcompare */
  0,                                          /* tp_weaklistoffset */
  nullptr,                                    /* tp_iter */
  nullptr,                                    /* tp_iternext */
  nullptr,
  /* will be assigned in init */ /* tp_methods */
  nullptr,
  /* will be assigned in init */ /* tp_members */
  nullptr,                       /* tp_getset */
  nullptr,                       /* tp_base */
  nullptr,                       /* tp_dict */
  nullptr,                       /* tp_descr_get */
  nullptr,                       /* tp_descr_set */
  0,                             /* tp_dictoffset */
  nullptr,                       /* tp_init */
  nullptr,                       /* tp_alloc */
  nullptr,                       /* tp_new */
};

void StoragePy_assertNotNull(StoragePy *storage) {
  if (StoragePy_Unpack(storage).get_storage_base() == nullptr) {
    MS_LOG(EXCEPTION) << "Got a null Storage";
  }
}

void StoragePy_assertNotNull(PyObject *obj) { StoragePy_assertNotNull(reinterpret_cast<StoragePy *>(obj)); }

static PyObject *StoragePy_Copy_(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  auto self_ = StoragePy_Unpack(self);
  PyObject *py_type;
  static const char *kwlist[] = {"src", "non_blocking", nullptr};
  int non_blocking = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", const_cast<char **>(kwlist), &py_type, &non_blocking)) {
    return nullptr;
  }

  auto src = StoragePy_Unpack(py_type);
  if (self_.NBytes() != src.NBytes()) {
    MS_LOG(EXCEPTION) << "Size does not match, self was " << self_.NBytes() << " bytes but src was " << src.NBytes()
                      << " bytes";
  }

  // ToDo: use tensor.copy_ to implement storage copy_
  self_.InplaceCopy(src, non_blocking);
  Py_INCREF(self);
  return self;

  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_ElementSize(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  return PyLong_FromLongLong(sizeof(uint8_t));
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_Resize_(PyObject *self, PyObject *number_arg) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  const auto &storage = StoragePy_Unpack(self);

  if (!(PyLong_CheckExact(number_arg) && !PyBool_Check(number_arg))) {
    MS_LOG(EXCEPTION) << "resize_ expects an int, but got " << Py_TYPE(number_arg)->tp_name;
  }

  int64_t newsize = PyLong_AsLong(number_arg);
  auto device_type = storage.device();
  if (device_type == "Ascend" || device_type == "CPU") {
    // ToDo: inplement resize
    storage.get_mutable_storage_base()->InplaceReSize(newsize);
  } else if (device_type == "GPU") {
    MS_LOG(EXCEPTION) << "Current Storage only support NPU, but got GPU!";
  }
  Py_INCREF(self);
  return self;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_NBytes(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  return py::cast(StoragePy_Unpack(self).NBytes()).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_DataPtr(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  auto self_ = StoragePy_Unpack(self);
  PyObject *dataPtrResult = PyLong_FromVoidPtr(reinterpret_cast<void *>(self_.DataPtr()));
  return dataPtrResult;
  HANDLE_MS_EXCEPTION_END
}

static PyMethodDef StoragePy_methods[] = {
  {"copy_", reinterpret_cast<PyCFunction>(StoragePy_Copy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"element_size", StoragePy_ElementSize, METH_NOARGS, nullptr},
  {"resize_", StoragePy_Resize_, METH_O, nullptr},
  {"nbytes", StoragePy_NBytes, METH_NOARGS, nullptr},
  {"size", StoragePy_NBytes, METH_NOARGS, nullptr},
  {"data_ptr", StoragePy_DataPtr, METH_NOARGS, nullptr},
  {nullptr}};

void RegStorage(py::module *m) {
  static std::vector<PyMethodDef> methods;
  size_t i = 0;
  while (true) {
    methods.push_back(StoragePy_methods[i]);
    if (!StoragePy_methods[i].ml_name) {
      break;
    }
    i++;
  }

  StoragePyType.tp_methods = methods.data();
  if (PyType_Ready(&StoragePyType) < 0) {
    return;
  }
  Py_INCREF(&StoragePyType);
  m->add_object("StoragePy", reinterpret_cast<PyObject *>(&StoragePyType));
}

PyObject *CreateStorageObj(const Storage &storage) {
  PyTypeObject *type = &StoragePyType;
  PyObject *obj = type->tp_alloc(type, 0);
  auto s = reinterpret_cast<StoragePy *>(obj);
  new (&s->cdata) Storage();
  s->cdata = std::move(storage);
  return obj;
}
}  // namespace mindspore
