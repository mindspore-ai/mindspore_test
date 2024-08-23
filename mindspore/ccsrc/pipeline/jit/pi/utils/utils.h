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
#ifndef MINDSPORE_PI_JIT_UTILS_H
#define MINDSPORE_PI_JIT_UTILS_H

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <chrono>
#include <algorithm>
#include "pybind11/pybind11.h"
#include "ir/cell.h"
#include "pipeline/jit/pi/utils/opcode_declare.h"
#include "pipeline/jit/pi/python_adapter/py_code.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

constexpr auto kTwo = 2;
constexpr auto kThree = 3;
constexpr auto kFive = 5;

enum StopTraceReason : uint8_t {
#define STOP_TRACE_REASON_KIND(kind, description) k##kind,
#include "stop_trace_reason.def"
#undef STOP_TRACE_REASON_KIND
};

std::string GetStopTraceReasonDesc(StopTraceReason res);

enum InlineReason : uint8_t {
#define INLINE_REASON_KIND(kind, description) k##kind,
#include "inline_reason.def"
#undef INLINE_REASON_KIND
};

std::string GetInlineReasonDesc(InlineReason res);

enum LoopUnrollingReason : uint8_t {
#define LOOP_UNROLLING_REASON_KIND(kind, description) k##kind,
#include "loop_unrolling_reason.def"
#undef LOOP_UNROLLING_REASON_KIND
};

std::string GetLoopUnrollingReasonDesc(LoopUnrollingReason res);

class Utils {
 public:
  Utils() = default;
  ~Utils() = default;

  static std::string GetPyName(PyObject *obj);

  static PyFrameObject *PrepareFrame(PyObject *callable, PyObject *args, PyObject *kwargs);

  // find a object from specified module. default not import, not throw.
  static py::object GetModuleAttr(const std::string &mod_name, const std::string &attr_name, bool _import = false,
                                  bool _throw = false);

  // if has a python exception, log it and return the exception information
  static std::string ReportPythonException();

  /**
   * Pack stack arguments to PyObject by opcode
   *
   * \param args stack arguments, the layout match opcode.
   * \param callop CALL_FUNCTION/CALL_METHOD/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
   * \param ret_vector_args if true, return a tuple arguments with names tuple.
   *                        default, return a tuple arguments with a dict arguments.
   *                        if failed, pair.first is empty.
   * \return a pair of arguments for object call
   */
  static std::pair<py::object, py::object> PackCallStackArgs(const std::vector<py::object> &args, int opcode,
                                                             bool ret_vector_args = false);

  // alias python 'print(func); import dis; dis.dis(func)'
  static void DisFuncObject(PyObject *);
  // alias python 'print(...)'
  static void PyBuiltinPrint(PyObject *);

  static PyObject *MixedPrecisionTypeToDType(MixedPrecisionType mixedPrecisionType);
};

#define GRAPH_JIT_LOG_F PY_PRINT_F

#define PY_PRINT_F(fmt, ...)                                       \
  do {                                                             \
    PyObject *_pystr;                                              \
    if (fmt[strlen(fmt) - 1] == '\n') {                            \
      std::string _fstr = fmt;                                     \
      _fstr[_fstr.size() - 1] = ' ';                               \
      _pystr = PyUnicode_FromFormat(_fstr.c_str(), ##__VA_ARGS__); \
    } else {                                                       \
      _pystr = PyUnicode_FromFormat(fmt, ##__VA_ARGS__);           \
    }                                                              \
    Utils::PyBuiltinPrint(_pystr);                                 \
    Py_DECREF(_pystr);                                             \
  } while (0)

#define REPLACE_PY_MEMBER(member, o)     \
  do {                                   \
    PyObject *py_replace_tmp = (member); \
    Py_XINCREF(o);                       \
    (member) = (o);                      \
    Py_XDECREF(py_replace_tmp);          \
  } while (0)

#ifdef DEBUG
#define PRINT_IF_HAS_USER_DEFINED_HOOK(op, hook)                                         \
  do {                                                                                   \
    static const char *slot_key_##hook = #hook;                                          \
    PyObject *attr_##hook = PyObject_GetAttrString(op, slot_key_##hook);                 \
    if (attr_##hook && (PyMethod_Check(attr_##hook) || PyFunction_Check(attr_##hook))) { \
      PY_PRINT_F("%A has hook " #hook, PyType_Check(op) ? op : (PyObject *)Py_TYPE(op)); \
    } else {                                                                             \
      PyErr_Clear();                                                                     \
    }                                                                                    \
    Py_XDECREF(attr_##hook);                                                             \
  } while (0)
#else
#define PRINT_IF_HAS_USER_DEFINED_HOOK(op, hook)
#endif
class ReprRecursionScope {
 public:
  explicit ReprRecursionScope(PyObject *v) : v_(v), stat_(v == nullptr ? -1 : Py_ReprEnter(v)) {}
  ~ReprRecursionScope() {
    if (stat_ == 0) {
      Py_ReprLeave(v_);
    }
  }
  bool ErrExist() { return stat_ < 0; }
  bool ReEnter() { return stat_ > 0; }
  bool ReEnterOrError() { return ReEnter() || ErrExist(); }

 private:
  PyObject *v_;
  int stat_;
};

bool HasMutableOrConstAttr(PyObject *obj);
bool IsMutableObj(const py::object &obj);
bool CheckMutableOrNonConstAttr(PyObject *obj);
bool HasDynamicLength(PyObject *obj);
bool CheckDynamicLength(PyObject *obj);
bool CheckScalar(PyObject *obj);
bool CheckContainer(PyObject *obj);
bool IsTensorPyObject(PyObject *obj);
bool IsMsClass(PyObject *obj);
bool IsNumpyObject(PyObject *obj);
bool IsNoGradEnterFunc(const py::object &handle);
bool IsNoGradExitFunc(const py::object &handle);
const char *GetFuncName(const py::object &handle);

bool CheckAdapterTensor(const py::object &tensor);
py::object ConvertToMsTensor(const py::object &tensor);
py::object ConvertToAdapterTensor(const py::object &tensor);

std::string GetTopModule(const py::object &o);
py::object GetPyCodeObject(const py::object &any, bool exact_func = false);
size_t DeviceAvailableMemSize();
bool CheckConstPyObject(PyObject *cnst);

template <typename T>
class PackCallStackHelper {
 public:
  struct Packed {
    std::vector<T> args_;
    std::map<std::string, T> kw_;
  };

  explicit PackCallStackHelper(int opcode) : opcode_(opcode) {}
  auto &result() { return result_; }

  template <typename CastKeys, typename CastSequence, typename CastKeyWords>
  bool Pack(const std::vector<T> &stack_args, CastKeys to_keys, CastSequence to_seq, CastKeyWords to_map) {
    if (opcode_ == CALL_FUNCTION_EX) {
      result_.args_ = to_seq(stack_args[0]);
      result_.kw_ = stack_args.size() > 1 ? to_map(stack_args[1]) : std::map<std::string, T>();
      return true;
    }
    if (opcode_ != CALL_FUNCTION_KW && opcode_ != CALL_FUNCTION) {
      return false;
    }
    size_t size = stack_args.size();
    if (opcode_ == CALL_FUNCTION_KW) {
      std::vector<std::string> keys = to_keys(stack_args.back());
      size = size - keys.size() - 1;
      for (size_t i = 0; i < keys.size(); ++i) {
        result_.kw_[std::move(keys[i])] = stack_args[size + i];
      }
    }
    std::copy(stack_args.begin(), stack_args.begin() + size, std::back_inserter(result_.args_));
    return true;
  }

 private:
  Packed result_;
  int opcode_;
};

template <typename E>
class BindArgumentsHelper {
 public:
  struct BindArguments {
    /**
     * args contains kw-only arguments, ordered by code.co_varnames
     * e.g:
     * def func(a,/,b,*va,c,**kw_va):
     *    pass
     * func.__code__.co_varnames = ('a', 'b', 'c', 'va', 'kw_va')
     * args = [a, b, c, va, kw_va]
     */
    std::vector<E> args_;
    std::vector<E> va_;
    std::map<std::string, E> kw_va_;
  };
  explicit BindArgumentsHelper(PyCodeObject *co) : co_(co) {
    results_.args_.resize(co->co_argcount + co->co_kwonlyargcount);
  }
  auto &results() { return results_; }

 private:
  PyCodeObject *co_;
  BindArguments results_;

 public:
  // return false if arguments not match, not check missing arguments
  bool Bind(const std::vector<E> &args, const std::map<std::string, E> &kw = {}) {
    const int argc = args.size();
    PyCodeWrapper wrapper(co_);
    const int position_only = wrapper.PositionOnlyArgCount();
    const int position_argc = std::min(argc, co_->co_argcount);
    const int parameter_argc = results_.args_.size();

    for (int i = 0; i < position_argc; i++) {
      results_.args_[i] = args[i];
    }
    if (co_->co_flags & CO_VARARGS) {
      results_.va_ = {args.begin() + position_argc, args.end()};
    } else if (argc > co_->co_argcount) {
      MS_LOG(ERROR) << "takes " << co_->co_argcount << " positional arguments but " << argc << " was given";
      return false;
    }
    PyObject **begin = &PyTuple_GET_ITEM(PyCodeWrapper(co_).VarNames().ptr(), 0);
    PyObject **end = begin + parameter_argc;
    for (const auto &item : kw) {
      const auto &k = item.first;
      const auto &v = item.second;
      auto iter = std::find_if(begin, end, [&k](PyObject *o) { return k == PyUnicode_AsUTF8(o); });
      auto index = iter - begin;
      if (iter == end && (co_->co_flags & CO_VARKEYWORDS) == 0) {
        MS_LOG(ERROR) << "got an unexpected keyword argument '" << k << "'";
        return false;
      }
      if (iter != end && (index < position_only || results_.args_[index] != E())) {
        MS_LOG(ERROR) << (index < position_only ? "missing 1 required positional argument: '"
                                                : "got multiple values for argument '")
                      << k << "'";
        return false;
      }
      if (iter != end) {
        results_.args_[index] = v;
      } else {
        results_.kw_va_.insert(item);
      }
    }
    return true;
  }

  // check no defaults arguments
  bool CheckArguments(size_t start, size_t stop) {
    auto begin = results_.args_.begin() + start;
    auto end = results_.args_.begin() + stop;
    auto iter = std::find(begin, end, E());
    if (iter == end) {
      return true;
    }
    auto index = iter - results_.args_.begin();
    MS_LOG(ERROR) << "missing 1 required positional argument at " << index;
    return false;
  }

  // return false if missing arguments
  template <typename Converter>
  bool ApplyDefault(PyObject *defaults, PyObject *kw_defaults, Converter convert) {
    defaults = defaults ? defaults : Py_None;
    kw_defaults = kw_defaults ? kw_defaults : Py_None;
    bool is_valid_defaults = (defaults == Py_None || PyTuple_Check(defaults));
    bool is_valid_kw_defaults = (kw_defaults == Py_None || PyDict_Check(kw_defaults));
    MS_EXCEPTION_IF_CHECK_FAIL(is_valid_defaults && is_valid_kw_defaults, "error arguments");
    const size_t defaults_size = defaults == Py_None ? 0 : PyTuple_GET_SIZE(defaults);
    const size_t off = co_->co_argcount - defaults_size;
    MS_EXCEPTION_IF_CHECK_FAIL(defaults_size <= static_cast<size_t>(co_->co_argcount), "error function defaults");
    if (!CheckArguments(0, off)) {
      return false;
    }
    auto varnames = PyCodeWrapper(co_).VarNames().ptr();
    for (size_t i = off, argc = results_.args_.size(); i < argc; ++i) {
      if (results_.args_[i] != E()) {
        continue;
      } else if (i - off < defaults_size) {
        results_.args_[i] = convert(defaults, py::int_(i - off).ptr(), PyTuple_GET_ITEM(defaults, i - off));
        continue;
      }
      auto k = PyTuple_GET_ITEM(varnames, i);
      auto v = PyDict_GetItem(kw_defaults, k);
      if (v == nullptr) {
        MS_LOG(ERROR) << "missing 1 required keyword-only argument: '" << PyUnicode_AsUTF8(k) << "'";
        return false;
      }
      results_.args_[i] = convert(kw_defaults, k, v);
    }
    return true;
  }
};

class TimeRecorder {
 public:
  using RecorderType = const char *;
  static constexpr double scale = std::nano::den;

  class TimeData {
   public:
    struct Data {
      uint64_t count;
      uint64_t nano;
    };
    TimeData() = default;
    ~TimeData();
    std::string ToString();

    std::map<RecorderType, Data> data_;
  };

  explicit TimeRecorder(const RecorderType &descr, bool record = true);
  ~TimeRecorder();

 private:
  static TimeData *Data();

  RecorderType descr_;
  std::chrono::steady_clock::time_point start_;
  bool record_;
};

class RefTracker {
 public:
  ~RefTracker();
  bool Track(PyObject *obj, const std::string &descr);
  static RefTracker *GetInstance();

 private:
  RefTracker();
  static PyObject *UnTrack(PyObject *ref, PyObject *);
  std::map<void *, std::pair<PyObject *, PyObject *>> tracked_;
  PyMethodDef mdef_;
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_UTILS_H
