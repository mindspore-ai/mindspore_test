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

#include <set>
#include "ms_extension.h"

using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;
using TypeId = mindspore::TypeId;
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

ShapeVector ReduceSumInferShape(const BaseTensorPtr &t, const std::vector<int64_t> &axis, bool keepdims) {
  auto &s = t->shape();
  size_t n = s.size();
  if (!axis.empty()) {
    ShapeVector ret;
    std::set<size_t> axis_set;
    for (auto x : axis) {
      axis_set.insert(static_cast<size_t>(x > 0 ? x : x + static_cast<int64_t>(n)));
    }
    for (size_t i = 0; i < n; i++) {
      if (axis_set.count(i) > 0) {
        if (keepdims) {
          ret.push_back(1LL);
        }
      } else {
        ret.push_back(s[i]);
      }
    }
    return ret;
  } else {
    return keepdims ? ShapeVector(n, 1LL) : ShapeVector();
  }
}

namespace mindspore {
BaseTensorPtr npu_reduce_sum(const BaseTensorPtr &x, const std::optional<std::vector<int64_t>> &axis, bool keepdims,
                             std::optional<int64_t> dtype) {
  mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                mindspore::runtime::ProfilerEvent::kRunOp, "npu_reduce_sum");
  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = mindspore::runtime::OpRunner::GetDeviceContext("Ascend");
  auto type_id = dtype.has_value() ? static_cast<TypeId>(dtype.value()) : x->data_type();
  auto axis_vec = axis.has_value() ? axis.value() : std::vector<int64_t>();
  auto result = std::make_shared<BaseTensor>(type_id, ReduceSumInferShape(x, axis_vec, keepdims));

  auto y = kernel::pyboost::abs(x);

  // create DeviceAddress for inputs and outputs
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, y);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, {result});

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([=]() {
    MS_LOG(DEBUG) << "Run device task aclnnReduceSum start";
    // malloc device memory for inputs and outputs
    PyBoostUtils::MallocOpInputs(device_context, y);
    PyBoostUtils::MallocOpOutputs(device_context, {result});
    // launch aclnn op
    LAUNCH_ACLNN(aclnnReduceSum, device_context, stream_id, y, axis_vec, keepdims, type_id, result);
    MS_LOG(DEBUG) << "Run device task aclnnReduceSum end";
  }));
  return result;
}
}  // namespace mindspore

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_reduce_sum", &mindspore::npu_reduce_sum, "aclnnReduceSum", pybind11::arg("x"),
        pybind11::arg("axis") = std::nullopt, pybind11::arg("keepdims") = false, pybind11::arg("dtype") = std::nullopt);
}
