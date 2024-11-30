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
#include <vector>
#include <string>

#include "include/common/pybind_api/api_register.h"
#include "include/backend/debug/data_dump/dump_control.h"
#include "include/backend/debug/data_dump/tensordump_control.h"
#include "utils/ms_context.h"

namespace py = pybind11;
namespace mindspore {
namespace dump {
void RegDumpControl(py::module *m) {
  m->def("_dump_set_dynamic", []() { DumpControl::GetInstance().SetDynamicDump(); })
    .def("_dump_start", []() { DumpControl::GetInstance().DynamicDumpStart(); })
    .def("_dump_stop", []() { DumpControl::GetInstance().DynamicDumpStop(); })
    .def("_tensordump_set_step",
         [](const std::vector<size_t> &v) { TensorDumpStepManager::GetInstance().SetDumpStep(v); })
    .def("_tensordump_process_file", [](const std::string &filename, const std::string &dtype) -> std::string {
      return TensorDumpStepManager::GetInstance().ProcessFileName(filename, dtype, kPynativeMode);
    });
}

}  // namespace dump

}  // namespace mindspore
