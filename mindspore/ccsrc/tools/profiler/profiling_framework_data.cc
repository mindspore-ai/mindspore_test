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
#include "tools/profiler/profiling_framework_data.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#include <utility>
#include <algorithm>
#include <mutex>
#include <numeric>
#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"

namespace mindspore {
namespace profiler {
namespace ascend {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void ProfilingFrameworkData::RecordHostProfile(std::shared_ptr<ProfilerData> data) {
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  if (!profiler_manager->EnableCollectHost()) {
    return;
  }
  auto &instance = runtime::ProfilerAnalyzer::GetInstance();
  std::unique_ptr<OpRangeData> report = std::make_unique<OpRangeData>(
    ProfilingFrameworkData::Device_Id, data->tid_, data->flow_id_, instance.step(), data->start_time_, data->end_time_,
    data->tid_, static_cast<uint16_t>(data->module_), static_cast<uint16_t>(data->event_),
    static_cast<uint16_t>(data->stage_), data->level_, data->is_graph_data_, data->is_stage_, data->op_name_,
    data->op_full_name_, data->module_graph_, data->event_graph_, data->custom_info_);
  ProfilingDataDumper::GetInstance().Report(std::move(report));
}

void ProfilingFrameworkData::RecordShapesProfile(const std::string &op_name,
                                                 const std::vector<std::vector<int64_t>> &input_shapes,
                                                 const std::vector<std::string> &input_types) {
  std::string input_shapes_str = "";
  for (auto &shape_vector : input_shapes) {
    for (auto &shape : shape_vector) {
      input_shapes_str.append(std::to_string(shape)).append(",");
    }
    if (!input_shapes_str.empty() && input_shapes_str.back() == ',') {
      input_shapes_str.pop_back();
    }
    input_shapes_str.append(";");
  }
  std::string input_type_str = std::accumulate(input_types.begin(), input_types.end(), std::string{},
                                               [](const std::string &a, const std::string &b) { return a + b + ';'; });
  if (!input_shapes_str.empty()) {
    input_shapes_str.pop_back();
  }
  if (!input_type_str.empty()) {
    input_type_str.pop_back();
  }
  std::unique_ptr<RecordShapesData> report =
    std::make_unique<RecordShapesData>(ProfilingFrameworkData::Device_Id, op_name, input_shapes_str, input_type_str);
  ProfilingDataDumper::GetInstance().Report(std::move(report));
}
#else
void ProfilingFrameworkData::RecordHostProfile(std::shared_ptr<ProfilerData> data) {
  MS_LOG(INTERNAL_EXCEPTION) << "host profiler not support cpu windows.";
}

void ProfilingFrameworkData::RecordShapesProfile(const std::string &op_name,
                                                 const std::vector<std::vector<int64_t>> &input_shapes,
                                                 const std::vector<std::string> &input_types) {
  MS_LOG(INTERNAL_EXCEPTION) << "shapes profiler not support cpu windows.";
}
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
