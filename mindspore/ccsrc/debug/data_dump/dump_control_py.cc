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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "include/backend/debug/data_dump/dump_control.h"

namespace py = pybind11;
namespace mindspore {

PYBIND11_MODULE(_data_dump, m) {
  m.def("_dump_set_dynamic", []() { DumpControl::GetInstance().SetDynamicDump(); });
  m.def("_dump_start", []() { DumpControl::GetInstance().DynamicDumpStart(); });
  m.def("_dump_stop", []() { DumpControl::GetInstance().DynamicDumpStop(); });
}

}  // namespace mindspore
