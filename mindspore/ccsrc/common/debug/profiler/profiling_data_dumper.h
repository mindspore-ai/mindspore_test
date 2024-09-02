/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_

#include <sys/stat.h>
#include <fcntl.h>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <libgen.h>
#include <linux/limits.h>
#include <sys/syscall.h>
#endif
#include <stdint.h>
#include <fstream>
#include <queue>
#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "utils/ms_utils.h"
#include "include/common/visible.h"

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr uint32_t kDefaultRingBuffer = 2000 * 1000;
constexpr uint32_t kBatchMaxLen = 5 * 1024 * 1024;  // 5 MB
constexpr uint32_t kMaxWaitTimeUs = 100 * 1000;
constexpr uint32_t kMaxWaitTimes = 10;

class COMMON_EXPORT Utils {
 public:
  static bool IsFileExist(const std::string &path);
  static bool IsFileWritable(const std::string &path);
  static bool IsDir(const std::string &path);
  static bool CreateDir(const std::string &path);
  static std::string RealPath(const std::string &path);
  static std::string RelativeToAbsPath(const std::string &path);
  static std::string DirName(const std::string &path);
  static uint64_t GetClockMonotonicRawNs();
  static bool CreateDumpFile(const std::string &path);
  static bool IsSoftLink(const std::string &path);
  static uint64_t GetTid();
  static uint64_t GetPid();
};

template <typename T>
class COMMON_EXPORT RingBuffer {
 public:
  RingBuffer()
      : is_inited_(false),
        is_quit_(false),
        read_index_(0),
        write_index_(0),
        idle_write_index_(0),
        capacity_(0),
        mask_(0) {}

  ~RingBuffer() { UnInit(); }
  void Init(size_t capacity);
  void UnInit();
  size_t Size();
  bool Push(T data);
  T Pop();
  bool Full();
  void Reset();

 private:
  bool is_inited_;
  volatile bool is_quit_;
  std::atomic<size_t> read_index_;
  std::atomic<size_t> write_index_;
  std::atomic<size_t> idle_write_index_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> data_queue_;
};

struct COMMON_EXPORT BaseReportData {
  int32_t device_id{0};
  std::string tag;
  BaseReportData(int32_t device_id, std::string tag) : device_id(device_id), tag(std::move(tag)) {}
  virtual ~BaseReportData() = default;
  virtual std::vector<uint8_t> encode() = 0;
  virtual void preprocess() = 0;
};

enum class COMMON_EXPORT OpRangeDataType {
  OP_RANGE_DATA = 1,
  IS_ASYNC = 2,
  NAME = 3,
  INPUT_DTYPES = 4,
  INPUT_SHAPE = 5,
  STACK = 6,
  MODULE_HIERARCHY = 7,
  EXTRA_ARGS = 8,
  RESERVED = 30,
};

struct COMMON_EXPORT OpRangeData : BaseReportData {
  int64_t start_ns{0};
  int64_t end_ns{0};
  int64_t sequence_number{0};
  uint64_t process_id{0};
  uint64_t start_thread_id{0};
  uint64_t end_thread_id{0};
  uint64_t forward_thread_id{0};
  bool is_async{false};
  std::string name;
  std::vector<std::string> input_dtypes;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::string> stack;
  std::vector<std::string> module_hierarchy;
  uint64_t flow_id{0};
  uint64_t step{0};
  OpRangeData(int64_t start_ns, int64_t end_ns, int64_t sequence_number, uint64_t process_id, uint64_t start_thread_id,
              uint64_t end_thread_id, uint64_t forward_thread_id, bool is_async, std::string name,
              std::vector<std::string> stack, uint64_t flow_id, int32_t device_id, uint64_t step)
      : BaseReportData(device_id, "op_range_" + std::to_string(device_id)),
        start_ns(start_ns),
        end_ns(end_ns),
        sequence_number(sequence_number),
        process_id(process_id),
        start_thread_id(start_thread_id),
        end_thread_id(end_thread_id),
        forward_thread_id(forward_thread_id),
        is_async(is_async),
        name(std::move(name)),
        stack(std::move(stack)),
        flow_id(flow_id),
        step(step) {}

  OpRangeData(int64_t start_ns, int64_t end_ns, uint64_t start_thread_id, std::string name, int32_t device_id)
      : BaseReportData(device_id, "op_range_" + std::to_string(device_id)),
        start_ns(start_ns),
        end_ns(end_ns),
        start_thread_id(start_thread_id),
        name(std::move(name)) {}

  std::vector<uint8_t> encode();
  void preprocess();
};

class COMMON_EXPORT ProfilingDataDumper {
 public:
  void Init(const std::string &path, int32_t rank_id, size_t capacity = kDefaultRingBuffer);
  void UnInit();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();
  void Flush();

  static ProfilingDataDumper &GetInstance();

 private:
  void Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap);
  void Run();
  void GatherAndDumpData();

 private:
  ProfilingDataDumper();
  virtual ~ProfilingDataDumper();

  std::string path_;
  int32_t rank_id_{0};
  std::atomic<bool> start_;
  std::atomic<bool> init_;
  std::atomic<bool> is_flush_{false};
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
  std::map<std::string, FILE *> fd_map_;
  std::mutex flush_mutex_;
  DISABLE_COPY_AND_ASSIGN(ProfilingDataDumper);
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
