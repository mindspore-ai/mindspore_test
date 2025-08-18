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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_TRACE_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_TRACE_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include "frontend/jit/pi/python_adapter/py_frame.h"
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/graph_guard/info.h"

namespace py = pybind11;

namespace mindspore {
namespace pijit {

// avoid call `py::object::operator==`
inline bool operator==(const py::object &p, PyObject *raw_p) { return p.ptr() == raw_p; }
inline bool operator==(PyObject *raw_p, const py::object &p) { return raw_p == p.ptr(); }
inline bool operator!=(const py::object &p, PyObject *raw_p) { return p.ptr() != raw_p; }
inline bool operator!=(PyObject *raw_p, const py::object &p) { return raw_p != p.ptr(); }

typedef enum _TraceType {
  Unknown = 0,
  Global,
  Deref,
  Closure,
  BuiltIn,
  Local,
  Param,
  Name,
  ClassDeref,
  Const,
  Item,
  Attr,
  Type,
  Operation,
  Customized,
  Unsupported,
} TraceType;

typedef struct _TraceContext {
  PyFrameWrapper frame_;
  // fast access cache
  PyCodeWrapper f_code_;
  py::object f_globals_;
  py::object f_builtins_;
  py::dict f_locals_;
} TraceContext, *PTraceContext;

class Trace : public std::enable_shared_from_this<Trace> {
 public:
  Trace(PyObject *obj, std::shared_ptr<Trace> origin);
  virtual ~Trace();
  virtual std::shared_ptr<Trace> GetOrigin();
  virtual TraceType GetTraceType() const;
  virtual TraceType GetOriginType();

  // return old trace of unique_cache if `this` is in the cache. or update cache and all trace referenced by this
  virtual std::shared_ptr<Trace> UniqueAll(std::map<size_t, std::shared_ptr<Trace>> *unique_cache);
  virtual bool operator==(const Trace &trace);
  virtual void Detach();

  // trace new object by python runtime
  virtual py::object Retrieve(PTraceContext context, bool perf = false) = 0;
  virtual std::string ToString(bool include_param = true) = 0;
  virtual void SimpleString(std::ostream *s) const;
  virtual std::string FormatString(std::map<Trace *, size_t> *cache);
  virtual const InfoPack &Info() = 0;
  virtual bool IsConst() const;
  virtual std::shared_ptr<Trace> Optimize();
  virtual std::shared_ptr<Trace> This();
  virtual void SetRelaxCount(int cnt);
  virtual int GetRelaxCount() const;
  virtual void EnableRelax();
  virtual bool RelaxEnabled() const;
  virtual bool IsSpecialized() const;
  virtual int GetDepth() const;

  void Cache(PTraceContext context, const py::object &obj);
  void ClearCache();
  PyObject *GetObject() const { return obj_.ptr(); }
  void SetObject(const py::handle &o) { obj_ = py::reinterpret_borrow<py::object>(o); }

 protected:
  py::object obj_;
  py::object retrieve_cache_;
  std::shared_ptr<Trace> origin_;
  TraceType originType_;
  TraceType curType_;
  std::string strTrace_;
  InfoPackPtr info_;
  int relax_count_;
  int relax_limit_;
  int depth_;
  bool is_const_;
  bool is_specialized_;
  bool retrieved_;
};
using TracePtr = std::shared_ptr<Trace>;
using TraceVector = std::vector<TracePtr>;

class RootTrace : public Trace {
 public:
  RootTrace(PyObject *obj, TraceType tt, int index = -1, std::string name = "", std::string module_name = "");
  virtual ~RootTrace() = default;
  virtual py::object Retrieve(PTraceContext context, bool perf = false);
  virtual std::string ToString(bool include_param = true);
  void SimpleString(std::ostream *s) const override;
  virtual void GetParam(int *index, std::string *name, std::string *module_name);
  virtual bool operator==(const Trace &trace);
  virtual const InfoPack &Info();
  static bool Support(TraceType tt);

 protected:
  py::object RetrieveGlobal(PTraceContext context);
  py::object RetrieveDeref(PTraceContext context);
  py::object RetrieveClosure(PTraceContext context);
  py::object RetrieveBuiltin(PTraceContext context);
  py::object RetrieveLocal(PTraceContext context);
  py::object RetrieveParam(PTraceContext context);
  py::object RetrieveName(PTraceContext context);
  py::object RetrieveClassDeref(PTraceContext context);

  int idx_;
  std::string name_;
  std::string module_name_;
};
using RootTracePtr = std::shared_ptr<RootTrace>;

class ConstTrace : public Trace {
 public:
  ConstTrace(PyObject *obj, int index);
  virtual ~ConstTrace() = default;
  virtual int GetIndex();
  virtual py::object Retrieve(PTraceContext context, bool perf = false);
  virtual std::string ToString(bool include_param = true);
  void SimpleString(std::ostream *s) const override;
  virtual bool operator==(const Trace &trace);
  virtual void Detach();
  virtual const InfoPack &Info();
  static bool Support(TraceType tt);

 protected:
  int index_;
};
using ConstTracePtr = std::shared_ptr<ConstTrace>;

class OpTrace : public Trace {
 public:
  OpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string name = "");
  virtual ~OpTrace() = default;
  int GetOpCode() const;
  int GetOpArgs() const;
  TracePtr GetParam(size_t idx) const;
  size_t GetParamCount() const;
  const std::string &GetName() const;

  std::shared_ptr<Trace> UniqueAll(std::map<size_t, std::shared_ptr<Trace>> *unique_cache) override;
  virtual py::object Retrieve(PTraceContext context, bool perf = false);
  virtual std::string ToString(bool include_param = true);
  virtual bool operator==(const Trace &trace);
  virtual void Detach();
  std::string FormatString(std::map<Trace *, size_t> *cache) override;
  virtual const InfoPack &Info();
  virtual TracePtr Optimize();
  virtual void SetRelaxCount(int cnt);
  static bool Support(TraceType tt);
  std::shared_ptr<Trace> Fold();

 protected:
  bool RetrieveParams(PTraceContext context, bool perf, std::vector<py::object> *p);
  void SimpleString(std::ostream *) const;
  void InitInfo();

  virtual void CheckSpecialize();
  virtual TracePtr RemoveCastDuplicatePatternPass();
  virtual TracePtr RemovePrimOutIsTensorPass();
  virtual TracePtr RemoveEmptyTensorPass();
  virtual void JudgeDTypeChangePass();
  virtual void JudgeDTypeScopePass();
  virtual void JudgeCodeChangePass();
  virtual void JudgeTrainFlagPass();
  virtual void JudgeCompareConstPass();
  virtual void JudgeContainsConstPass();
  virtual void JudgeInplaceAddConstPass();
  virtual void JudgeIsConstPass();
  virtual void JudgeBoundMethodPass();
  virtual void JudgeSubScrRandPass();
  virtual void JudgeDTypeTensorAttrPass();
  virtual void JudgeRelaxGuardFuncPass();

 protected:
  int opcode_;
  int opargs_;
  TraceVector params_;
  std::string name_;
  bool is_fold_;
};
using OpTracePtr = std::shared_ptr<OpTrace>;
TracePtr CreateOpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, const std::string &module_name = "",
                       const std::string &name = "", bool strict = false, bool print = false);

/// \brief retrieve the PyObject with ref count plus 1 which will be minus outside
typedef std::function<PyObject *(PTraceContext context)> RetrieveFunc;
typedef std::function<std::string(bool)> ToStringFunc;
class CustomizedTrace : public Trace {
 public:
  CustomizedTrace(PyObject *obj, RetrieveFunc rfunc, ToStringFunc sfunc);
  virtual ~CustomizedTrace() = default;
  virtual py::object Retrieve(PTraceContext context, bool perf = false);
  virtual std::string ToString(bool include_param = true);
  virtual const InfoPack &Info();
  static bool Support(TraceType tt);

 protected:
  RetrieveFunc retrieve_;
  ToStringFunc tostring_;
};
using CustomizedTracePtr = std::shared_ptr<CustomizedTrace>;

class UnsupportedTrace : public Trace {
 public:
  UnsupportedTrace(PyObject *obj, TraceVector params, int op, int arg);
  virtual ~UnsupportedTrace() = default;
  virtual py::object Retrieve(PTraceContext context, bool perf = false);
  virtual std::string ToString(bool include_param = true);
  virtual TraceVector GetParams();
  virtual void Detach();
  std::string FormatString(std::map<Trace *, size_t> *cache) override;
  virtual const InfoPack &Info();
  virtual void SetRelaxCount(int cnt);
  static bool Support(TraceType tt);

 protected:
  TraceVector params_;
  int op_;
  int arg_;
};
using UnsupportedTracePtr = std::shared_ptr<UnsupportedTrace>;

py::object GetObjectFromTrace(PyFrameWrapper frame, TracePtr trace);
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_TRACE_H
