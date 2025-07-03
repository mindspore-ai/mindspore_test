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

#include "include/common/utils/pyobj_manager.h"

namespace mindspore {
PyObjManager &PyObjManager::Get() {
  static PyObjManager manager;
  return manager;
}

void PyObjManager::Clear() {
  Py_XDECREF(tensor_module_);
  Py_XDECREF(abc_module_);
}

PyObject *PyObjManager::GetTensorModule() {
  if (tensor_module_ == nullptr) {
    tensor_module_ = PyImport_ImportModule("mindspore.common.tensor");
  }
  return tensor_module_;
}

PyObject *PyObjManager::GetAbcModule() {
  if (abc_module_ == nullptr) {
    abc_module_ = PyImport_ImportModule("abc");
  }
  return abc_module_;
}
}  // namespace mindspore
