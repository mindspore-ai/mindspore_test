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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <utility>
#include "kernel/kernel.h"
#include "common/debug/profiler/profiling_data_dumper.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace profiler {
namespace ascend {
using mindspore::runtime::kProfilerEventString;
using mindspore::runtime::kProfilerModuleString;
using mindspore::runtime::kProfilerStageString;
using mindspore::runtime::ProfilerData;

class COMMON_EXPORT ProfilingFrameworkData {
 public:
  static void RecordHostProfile(std::shared_ptr<ProfilerData> data);

  inline static std::map<std::string, uint64_t> kernel_launch_begin_;
  inline static int32_t Device_Id{0};
  inline static bool added{false};
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_
