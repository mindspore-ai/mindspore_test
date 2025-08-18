/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_

#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "include/backend/visible.h"
#include "include/common/visible.h"

namespace py = pybind11;
namespace mindspore {
FRONTEND_EXPORT void RegTyping(py::module *m);
FRONTEND_EXPORT void RegCNode(const py::module *m);
FRONTEND_EXPORT void RegCell(const py::module *m);
FRONTEND_EXPORT void RegMetaFuncGraph(const py::module *m);
FRONTEND_EXPORT void RegFuncGraph(const py::module *m);
FRONTEND_EXPORT void RegUpdateFuncGraphHyperParams(py::module *m);
FRONTEND_EXPORT void RegParamInfo(const py::module *m);
FRONTEND_EXPORT void RegPrimitive(const py::module *m);
FRONTEND_EXPORT void RegPrimitiveFunction(const py::module *m);
FRONTEND_EXPORT void RegFunctional(const py::module *m);
FRONTEND_EXPORT void RegSignatureEnumRW(const py::module *m);
FRONTEND_EXPORT void RegValues(const py::module *m);
void RegMsContext(const py::module *m);
void RegDeviceManagerConf(const py::module *m);
void RegSecurity(py::module *m);
void RegForkUtils(py::module *m);
void RegRandomSeededGenerator(py::module *m);
void RegStress(py::module *m);
void RegSendRecv(py::module *m);
void RegResetParams(py::module *m);
void RegCleanTdtChannel(py::module *m);
void RegTFT(py::module *m);
FRONTEND_EXPORT void RegTensorDoc(py::module *m);
void RegReuseDataPtr(py::module *m);
void RegPreJit(py::module *m);
void RegStorage(py::module *m);

namespace hal {
void RegStream(py::module *m);
void RegEvent(py::module *m);
FRONTEND_EXPORT void RegCommHandle(py::module *m);
void RegMemory(py::module *m);
void RegUtils(py::module *m);
}  // namespace hal
namespace initializer {
void RegRandomNormal(py::module *m);
}

namespace runtime {
BACKEND_EXPORT void RegRuntimeConf(py::module *m);
}  // namespace runtime

namespace pynative {
FRONTEND_EXPORT void RegPyNativeExecutor(const py::module *m);
FRONTEND_EXPORT void RegisterPyBoostFunction(py::module *m);
FRONTEND_EXPORT void RegisterCustomizeFunction(py::module *m);
FRONTEND_EXPORT void RegisterCellBackwardHookFunction(py::module *m);
FRONTEND_EXPORT void RegisterDetachFunction(py::module *m);
FRONTEND_EXPORT void RegisterFunctional(py::module *m);
namespace distributed {
FRONTEND_EXPORT void RegReducer(py::module *m);
}
}  // namespace pynative

namespace pynative::autograd {
FRONTEND_EXPORT void RegBackwardFunction(py::module *m);
FRONTEND_EXPORT void RegBackwardNode(py::module *m);
}  // namespace pynative::autograd

namespace pijit {
FRONTEND_EXPORT void RegPIJitInterface(py::module *m);
}

namespace tensor {
FRONTEND_EXPORT void RegMetaTensor(const py::module *m);
FRONTEND_EXPORT void RegCSRTensor(const py::module *m);
FRONTEND_EXPORT void RegCOOTensor(const py::module *m);
FRONTEND_EXPORT void RegRowTensor(const py::module *m);
FRONTEND_EXPORT void RegMapTensor(const py::module *m);
FRONTEND_EXPORT void RegPyTensor(py::module *m);
}  // namespace tensor

namespace profiler {
void RegProfilerManager(const py::module *m);
void RegProfiler(const py::module *m);
}  // namespace profiler

namespace datadump {
void RegDataDump(py::module *m);
}

namespace silentdetect {
void RegSilentDetect(py::module *m);
}

namespace prim {
FRONTEND_EXPORT void RegCompositeOpsGroup(const py::module *m);
}
#ifdef _MSC_VER
namespace abstract {
FRONTEND_EXPORT void RegPrimitiveFrontEval();
}
#endif

namespace ops {
void RegOpEnum(py::module *m);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_
