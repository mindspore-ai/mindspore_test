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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_

#include <utility>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <limits>
#include <Python.h>
#include "include/common/utils/python_adapter.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace profiler {
namespace py = pybind11;

constexpr size_t max_py_threads = std::numeric_limits<uint8_t>::max() + 1;

enum class COMMON_EXPORT Command { kStartOne = 0, kStartAll, kStop, kClear };

enum class COMMON_EXPORT TraceTag { kPy_Call = 0, kPy_Return, kC_Call, kC_Return };

struct COMMON_EXPORT TraceContext {
  PyObject_HEAD PyThreadState *thread_state_;
};

struct COMMON_EXPORT PythonFuncCallData {
  uint64_t start_time_{0};
  uint64_t end_time_{0};
  uint32_t map_index_{0};
  PythonFuncCallData(uint64_t start_time, uint64_t end_time, uint32_t map_index)
      : start_time_{start_time}, end_time_{end_time}, map_index_{map_index} {}
};

struct COMMON_EXPORT RawEvent {
  RawEvent(TraceTag tag, PyFrameObject *frame) : tag_(tag), frame_(frame), t_(0), misc_() {}

  RawEvent(TraceTag tag, PyFrameObject *frame, PyObject *arg) : RawEvent(tag, frame) { misc_.arg_ = arg; }

  TraceTag tag_{};
  PyFrameObject *frame_{nullptr};
  uint64_t t_{0};
  union {
    PyObject *arg_;  // kC_Call
    void *null_;     // Unused (placeholder), kPy_Call, kPy_Return, kC_Return
  } misc_{};

  uint8_t tag() const { return static_cast<uint8_t>(tag_); }

  std::string get_func_name() const {
    if (tag_ == TraceTag::kC_Call) {
      return py::repr(misc_.arg_);
    } else if (tag_ == TraceTag::kPy_Call) {
      auto line_no = std::to_string(frame_->f_code->co_firstlineno);
      auto file_name = py::cast<std::string>(frame_->f_code->co_filename);
      auto func_name = py::cast<std::string>(frame_->f_code->co_name);
      std::stringstream name_stream;
      name_stream << file_name << "(" << line_no << "): " << func_name;
      return name_stream.str();
    }
    return "";
  }
};

class COMMON_EXPORT PythonTracer final {
 public:
  static void call(Command c, uint32_t rank_id);
  static int pyProfileFn(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg);

 private:
  PythonTracer();
  static PythonTracer &singleton();

  void start(size_t max_threads = max_py_threads, uint32_t rank_id = 0);
  void stop();
  void clear();
  void recordPyCall(TraceContext *ctx, PyFrameObject *frame);
  void recordCCall(TraceContext *ctx, PyFrameObject *frame, PyObject *arg);
  void recordReturn(TraceContext *ctx, PyFrameObject *frame, PyObject *arg, TraceTag tag);
  void trackModule(PyFrameObject *frame);
  void reportPythonModuleCallDataToNpuProfiler(PyObject *mod_class, uint64_t idx);
  void reportPythonFuncCallDataToNpuProfiler(const RawEvent &event);
  bool starts_with(const std::string &str, const std::string &start);
  void Flush();

  bool active_{false};
  PyObject *module_call_code_{nullptr};
  std::vector<TraceContext *> trace_contexts_;
  std::mutex flush_mutex_;
  std::stack<uint64_t> call_syscnt_;
  uint64_t tid_{0};
  uint32_t rank_id_{0};
  uint64_t stack_cnt{0};
  std::deque<std::string> sys_path_list;
  std::deque<std::unique_ptr<PythonFuncCallData>> data_chunk_buf_;
  std::unordered_map<std::string, uint32_t> op_map_;
  std::atomic<uint32_t> op_index_{0};
  uint32_t max_call_data_count_{20 * 1000 * 1000};
};
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_
