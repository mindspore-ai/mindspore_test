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
#include <vector>
#include <string>

#include "include/common/pybind_api/api_register.h"
#include "debug/checksum/checksum_mgr.h"

namespace py = pybind11;
namespace mindspore {
namespace checksum {
void RegCheckSum(py::module *m) {
  m->def(
     "sdc_detect_start", []() { CheckSumMgr::GetInstance().CheckSumStart(); }, "Start SDC detect")
    .def(
      "sdc_detect_stop", []() { CheckSumMgr::GetInstance().CheckSumStop(); }, "Stop SDC detect")
    .def(
      "get_sdc_detect_result", []() { return CheckSumMgr::GetInstance().GetCheckSumResult(); },
      "Get SDC detect result");
}
}  // namespace checksum
}  // namespace mindspore
