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
#include "tools/profiler/profiling_data_dumper.h"
#include <algorithm>
#include <mutex>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/utils.h"

namespace mindspore {
namespace profiler {
namespace ascend {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
ProfilingDataDumper::ProfilingDataDumper() : name_("ProfilingDataDumper"), path_(""), start_(false), init_(false) {}

ProfilingDataDumper &ProfilingDataDumper::GetInstance() {
  static ProfilingDataDumper instance;
  return instance;
}
ProfilingDataDumper::~ProfilingDataDumper() { UnInit(); }

void ProfilingDataDumper::Init(const std::string &path, size_t capacity) {
  path_ = path;
  data_chunk_buf_.Init(capacity);
  init_.store(true);
  MS_LOG(INFO) << "Init ProfilingDataDumper, path: " << path << ", capacity: " << capacity;
}

void ProfilingDataDumper::UnInit() {
  if (init_.load()) {
    data_chunk_buf_.UnInit();
    init_.store(false);
    start_.store(false);
    for (auto &f : fd_map_) {
      if (f.second != nullptr) {
        fclose(f.second);
        f.second = nullptr;
      }
    }
    fd_map_.clear();
    MS_LOG(INFO) << "UnInit ProfilingDataDumper";
  }
}

void ProfilingDataDumper::Start() {
  if (!init_.load() || Thread::Start() != 0) {
    return;
  }
  start_.store(true);
  MS_LOG(INFO) << "Start ProfilingDataDumper";
}

void ProfilingDataDumper::Stop() {
  if (start_.load() == true) {
    start_.store(false);
    Thread::Stop();
    MS_LOG(INFO) << "ProfilingDataDumper Thread Stop";
  }
  Flush();
  MS_LOG(INFO) << name_ << " Dump finished, total size: " << dump_count_ << " bytes";
}

void ProfilingDataDumper::GatherAndDumpData() {
  std::unordered_map<std::string, std::vector<uint8_t>> dataMap;
  uint64_t batchSize = 0;
  while (batchSize < kBatchMaxLen) {
    std::unique_ptr<BaseReportData> data{nullptr};
    if (UNLIKELY(!data_chunk_buf_.Pop(data) || data == nullptr)) {
      break;
    }
    std::vector<uint8_t> encodeData = data->encode();
    ++batchSize;
    const std::string &key = kReportFileTypeMap.at(static_cast<ReportFileType>(data->tag));
    auto iter = dataMap.find(key);
    if (iter == dataMap.end()) {
      dataMap.insert({key, encodeData});
    } else {
      iter->second.insert(iter->second.end(), encodeData.cbegin(), encodeData.cend());
    }
  }
  if (dataMap.size() > 0) {
    static bool create_flag = true;
    if (create_flag) {
      create_flag = !Utils::CreateDir(this->path_);
    }
    Dump(dataMap);
  }
}

void ProfilingDataDumper::Run() {
  for (;;) {
    if (!start_.load()) {
      break;
    }
    if (data_chunk_buf_.Size() > kNotifyInterval) {
      GatherAndDumpData();
    } else {
      usleep(kMaxWaitTimeUs);
    }
  }
}

void ProfilingDataDumper::Flush() {
  while (data_chunk_buf_.Size() != 0) {
    GatherAndDumpData();
  }
}

void ProfilingDataDumper::Report(std::unique_ptr<BaseReportData> data) {
  if (UNLIKELY(!start_.load() || data == nullptr)) {
    return;
  }
  data_chunk_buf_.Push(std::move(data));
}

void ProfilingDataDumper::Dump(const std::unordered_map<std::string, std::vector<uint8_t>> &dataMap) {
  for (auto &data : dataMap) {
    FILE *fd = nullptr;
    const std::string dump_file = path_ + "/" + data.first;
    auto iter = fd_map_.find(dump_file);
    if (iter == fd_map_.end()) {
      if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
        continue;
      }
      fd = fopen(dump_file.c_str(), "ab");
      if (fd == nullptr) {
        continue;
      }
      fd_map_.insert({dump_file, fd});
    } else {
      fd = iter->second;
    }
    MS_LOG(INFO) << name_ << " Dump file path: " << dump_file << ", size: " << data.second.size();
    fwrite(reinterpret_cast<const char *>(data.second.data()), sizeof(char), data.second.size(), fd);
    dump_count_ += data.second.size();
  }
}
#else
ProfilingDataDumper::ProfilingDataDumper() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
ProfilingDataDumper::~ProfilingDataDumper() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
ProfilingDataDumper &ProfilingDataDumper::GetInstance() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
void ProfilingDataDumper::Init(const std::string &path, size_t capacity) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
void ProfilingDataDumper::UnInit() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::Start() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::Stop() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::Run() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::Flush() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::GatherAndDumpData() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
void ProfilingDataDumper::Report(std::unique_ptr<BaseReportData> data) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
void ProfilingDataDumper::Dump(const std::unordered_map<std::string, std::vector<uint8_t>> &dataMap) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
