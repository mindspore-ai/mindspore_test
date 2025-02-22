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
#ifndef MINDSPORE_PI_JIT_GUARD_UTILS_H
#define MINDSPORE_PI_JIT_GUARD_UTILS_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <tuple>
#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/graph_guard/trace.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {

typedef enum _GIType {
  GTUnknown = 0,
  GTType,
  GTRepr,
  GTAttr,
  GTEqual,
  GTId,
  kMatchIDS,
} GIType;

class GuardItem : public std::enable_shared_from_this<GuardItem> {
 public:
  explicit GuardItem(TracePtr var);
  virtual ~GuardItem() = default;
  virtual bool Check(const EvalFrameObject *frame, std::map<size_t, PyObject *> *cache = nullptr,
                     bool perf = false) = 0;
  virtual bool Check(PyObject *obj) = 0;
  virtual std::string ToString() = 0;
  virtual const InfoPack &Info() = 0;
  virtual void Replace(TracePtr dst, TracePtr src);
  virtual TracePtr GetTrace() const;
  virtual bool operator==(const GuardItem &obj) const;
  virtual GIType GetType() const { return type_; }
  virtual bool MatchDynamicShape(std::shared_ptr<GuardItem> other) { return false; }
  virtual PyObject *ApplyDynamicShape(PyObject *obj) { return nullptr; }
  virtual std::shared_ptr<GuardItem> Optimize();
  virtual std::shared_ptr<GuardItem> This() { return shared_from_this(); }
  int fail_count() const { return fail_count_; }
  void set_faile_count(int c) { fail_count_ = c; }

 protected:
  TracePtr var_;
  GIType type_;
  InfoPackPtr info_;
  std::string strGuard_;
  int fail_count_;  // retrieve_count_ same as call count
};
using GuardItemPtr = std::shared_ptr<GuardItem>;

/// \brief check whether elements are equal
/// \param[in] obj
/// \param[in] needSpecialize to check the content of buffer
/// \param[in] recurseDepth to check the hierarchy element access like a.b.c by depth
GuardItemPtr GuardEqual(TracePtr obj, bool needSpecialize = true, int recurseDepth = INT_MAX);
GuardItemPtr GuardType(TracePtr obj);
GuardItemPtr GuardId(TracePtr obj);
GuardItemPtr GuardAttr(TracePtr obj);
GuardItemPtr GuardRepr(TracePtr obj);
GuardItemPtr GuardIDS(const TracePtr &tv, const GuardItemPtr &reused);
bool IsPyObjectEqual(PyObject *src, PyObject *dst);
PyObject *GetMsModule();
PyObject *GetMsType();
PyObject *GetMsTensorType();

ValuePtr ScalarToDstDtypeValue(const ValuePtr &src_value, const std::pair<TypeId, bool> &dst_type);
tensor::BaseTensorPtr TensorToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GUARD_UTILS_H
