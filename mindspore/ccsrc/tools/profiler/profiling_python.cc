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
#include "tools/profiler/profiling_python.h"
#include <Python.h>
#include <typeinfo>
#include "tools/profiler/profiling_data_dumper.h"
#include "tools/profiler/profiling_framework_data.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/report_data.h"

namespace mindspore {
namespace profiler {
using mindspore::profiler::ascend::OpRangeData;
using mindspore::profiler::ascend::ProfilingDataDumper;

PythonTracer &PythonTracer::singleton() {
  static PythonTracer singleton_;
  return singleton_;
}

PythonTracer::PythonTracer() : active_(false) {
  pybind11::gil_scoped_acquire gil;
  module_call_code_ = py::module::import("mindspore.nn").attr("Cell").attr("__call__").attr("__code__").ptr();
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void PythonTracer::start(size_t max_threads, uint32_t rank_id) {
  MS_LOG(INFO) << "PythonTracer stop.";
  rank_id_ = rank_id;
  tid_ = LongToUlong(syscall(SYS_gettid));
  py::gil_scoped_acquire gil;
  std::vector<PyThreadState *> thread_states;
  thread_states.reserve(max_threads);  // 预分配空间以提高性能
  thread_states.push_back(PyThreadState_Get());

  if (max_threads > 1) {
    for (auto thread_state = thread_states[0]; thread_state != nullptr && thread_states.size() < max_threads;
         thread_state = PyThreadState_Next(thread_state)) {
      if (thread_state != thread_states[0]) {
        thread_states.push_back(thread_state);
      }
    }

    if (thread_states.size() == max_threads && PyThreadState_Next(thread_states.back()) != nullptr) {
      MS_LOG(WARNING) << "Only tracing first " << max_threads
                      << " threads. Total active threads: " << thread_states.size();
    }
  }

  const size_t STACK_MAX_DEPTH = 128;
  for (uint32_t i = 0; i < thread_states.size(); i++) {
    PyThreadState *thread_state = thread_states[i];
    PyThreadState_Swap(thread_state);
    std::vector<PyFrameObject *> current_stack;
    auto frame = PythonCApi::PyEval_GetFrame_MS();
    size_t depth = 0;  // Make sure we can't infinite loop.
    while (frame != nullptr && depth <= STACK_MAX_DEPTH) {
      current_stack.push_back(frame);
      frame = PythonCApi::PyFrame_GetBack_MS(frame);
      ++depth;
    }
    MS_LOG(INFO) << "start depth: " << depth;
    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyCall(nullptr, *it);
    }

    PyEval_SetProfile(PythonTracer::pyProfileFn, nullptr);
  }
  PyThreadState_Swap(thread_states[0]);
  auto paths = py::list(py::module::import("sys").attr("path"));
  for (auto &p : paths) {
    std::string p_str = py::cast<std::string>(p);
    if (p_str.empty()) {
      continue;
    }
    if (p_str[p_str.size() - 1] == '/') {
      sys_path_list.emplace_front(p_str);
    } else {
      sys_path_list.emplace_front(p_str + "/");
    }
  }
  active_ = true;
}

void PythonTracer::stop() {
  MS_LOG(INFO) << "PythonTracer stop.";
  if (active_ == false) {
    return;
  }
  py::gil_scoped_acquire gil;
  PyEval_SetProfile(nullptr, nullptr);
  active_ = false;
  Flush();
  op_map_.clear();
  while (!call_syscnt_.empty()) {
    call_syscnt_.pop();
  }
}

void PythonTracer::Flush() {
  MS_LOG(INFO) << "python stack count: " << data_chunk_buf_.size() << ", op_name: " << op_map_.size()
               << ", call_syscnt_: " << call_syscnt_.size();
  std::lock_guard<std::mutex> flush_lock(flush_mutex_);
  std::unordered_map<uint32_t, std::string> op_index_map;
  for (const auto &kv : op_map_) {
    std::string op_name = kv.first;
    uint32_t op_name_len = op_name.size();
    for (std::string &p : sys_path_list) {
      uint32_t path_len = p.size();
      if (op_name_len >= path_len && op_name.substr(0, path_len) == p) {
        op_name = op_name.substr(path_len, op_name_len - path_len);
      }
    }
    op_index_map.insert(std::make_pair(kv.second, std::move(op_name)));
  }
  for (auto &data : data_chunk_buf_) {
    std::unique_ptr<OpRangeData> op_range =
      std::make_unique<OpRangeData>(tid_, data->start_time_, data->end_time_, op_index_map[data->map_index_], rank_id_);
    ProfilingDataDumper::GetInstance().Report(std::move(op_range));
  }
  data_chunk_buf_.clear();
}

void PythonTracer::clear() {
  for (auto i : trace_contexts_) {
    Py_DECREF(reinterpret_cast<PyObject *>(i));
  }
  trace_contexts_.clear();
}

void PythonTracer::recordPyCall(TraceContext *ctx, PyFrameObject *frame) {
  call_syscnt_.push(profiler::GetClockSyscnt());
}

void PythonTracer::recordCCall(TraceContext *ctx, PyFrameObject *frame, PyObject *arg) {
  call_syscnt_.push(profiler::GetClockSyscnt());
}

void PythonTracer::recordReturn(TraceContext *ctx, PyFrameObject *frame, PyObject *arg, TraceTag tag) {
  if (call_syscnt_.empty()) {
    MS_LOG(WARNING) << "python stack is empty";
    return;
  }
  MS_EXCEPTION_IF_NULL(frame);
  uint64_t end_time = profiler::GetClockSyscnt();
  uint64_t start_time = call_syscnt_.top();
  call_syscnt_.pop();
  std::string op_name;
  if (tag == TraceTag::kPy_Return) {
    std::string py_class_name;
    auto f_code = PythonCApi::PyFrame_GetCode_MS(frame).get();
    MS_EXCEPTION_IF_NULL(f_code);
    if (reinterpret_cast<PyObject *>(f_code) == module_call_code_) {
      PyFrame_FastToLocals(frame);
      auto f_locals = reinterpret_cast<PyObject *>(PythonCApi::PyFrame_GetLocals_MS(frame).get());
      if (f_locals != nullptr) {
        auto module_class = PyDict_GetItemString(f_locals, "self");
        py_class_name = "nn.Cell." +
                        py::cast<std::string>(py::str(py::handle(module_class).attr("__class__").attr("__name__"))) +
                        ".";
      }
    }
    op_name = py::cast<std::string>(f_code->co_filename) + "(" + std::to_string(f_code->co_firstlineno) +
              "):" + py_class_name + py::cast<std::string>(f_code->co_name);
  } else if (arg != nullptr) {
    op_name = py::repr(arg);
  }
  uint32_t index;
  auto iter = op_map_.find(op_name);
  if (iter != op_map_.end()) {
    index = iter->second;
  } else {
    index = op_index_.fetch_add(1, std::memory_order_acquire);
    op_map_.insert(std::make_pair(std::move(op_name), index));
  }
  std::unique_ptr<PythonFuncCallData> call_data = std::make_unique<PythonFuncCallData>(start_time, end_time, index);
  data_chunk_buf_.emplace_front(std::move(call_data));
  if (data_chunk_buf_.size() > max_call_data_count_) {
    Flush();
  }
}

int PythonTracer::pyProfileFn(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
  auto ctx = reinterpret_cast<TraceContext *>(obj);
  switch (what) {
    case PyTrace_C_CALL:
      PythonTracer::singleton().recordCCall(ctx, frame, arg);
      break;

    case PyTrace_CALL:
      PythonTracer::singleton().recordPyCall(ctx, frame);
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      PythonTracer::singleton().recordReturn(ctx, frame, arg, TraceTag::kC_Return);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      PythonTracer::singleton().recordReturn(ctx, frame, arg, TraceTag::kPy_Return);
      break;

    default:
      break;
  }
  return 0;
}

void PythonTracer::call(Command cmd, uint32_t rank_id) {
  MS_LOG(INFO) << "PythonTracer Command: " << cmd;
  switch (cmd) {
    case Command::kStartAll:
      PythonTracer::singleton().start();
      break;

    case Command::kStartOne:
      PythonTracer::singleton().start(1, rank_id);
      break;

    case Command::kStop:
      PythonTracer::singleton().stop();
      break;

    case Command::kClear:
      PythonTracer::singleton().clear();
      break;

    default:
      break;
  }
}
#else
void PythonTracer::start(size_t max_threads, uint32_t rank_id) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void PythonTracer::stop() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void PythonTracer::Flush() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void PythonTracer::clear() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void PythonTracer::recordPyCall(TraceContext *ctx, PyFrameObject *frame) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void PythonTracer::recordCCall(TraceContext *ctx, PyFrameObject *frame, PyObject *arg) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void PythonTracer::recordReturn(TraceContext *ctx, PyFrameObject *frame, PyObject *arg, TraceTag tag) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

int PythonTracer::pyProfileFn(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void PythonTracer::call(Command c, uint32_t rank_id) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
#endif
}  // namespace profiler
}  // namespace mindspore
