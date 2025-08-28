/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_DATA_CONVERTER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_DATA_CONVERTER_H_

#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <Python.h>

#include "utils/ordered_map.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "ops/op_def.h"
#include "include/common/visible.h"

namespace mindspore {
namespace parse {
// data convert for parse
namespace data_converter {
void CacheObjectValue(const std::string &obj_key, const ValuePtr &data);
bool GetObjectValue(const std::string &obj_key, ValuePtr *const data);

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data);

const mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs();

std::vector<std::string> GetObjKey(const py::object &obj);
ResolveType GetObjType(const py::object &obj);
ClassInstanceType GetClassInstanceType(const py::object &obj);

bool IsCellInstance(const py::object &obj);
bool IsNumpyArrayInstance(const py::object &obj);
bool IsMsClassInstance(const py::object &obj);
bool IsJITForbiddenAPI(const py::object &obj);
bool IsClassType(const py::object &obj);
py::object CreatePythonObject(const py::object &type, const py::tuple &args_kwargs);
FRONTEND_EXPORT py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs);
py::set GetPythonScriptIdAttrs(const py::object &script);
void MakeProperNameToFuncGraph(const FuncGraphPtr &func_graph, std::string name);
FRONTEND_EXPORT ValuePtr PyDataToValue(const py::object &obj);
ValuePtr PyDataToStubNode(const py::object &obj);
FRONTEND_EXPORT void ClearObjectCache();
FRONTEND_EXPORT ValuePtr PyObjToValue(const py::object &obj, bool stub = false);
}  // namespace data_converter

class DataConverter {
 public:
  DataConverter(ValuePtrList args_value_list, bool use_signature)
      : args_value_list_(std::move(args_value_list)),
        use_signature_(use_signature),
        dtype_(nullptr),
        forbid_reuse_(false) {}

  virtual ~DataConverter() = default;

  ValuePtr ConvertData(const py::object &obj);

 private:
  ValuePtrList args_value_list_;
  bool use_signature_;
  TypePtr dtype_;
  bool forbid_reuse_;
};

FuncGraphPtr ConvertToBpropCut(const py::object &obj);
constexpr int32_t kTypeShiftBits = 16;
constexpr auto kDstMask = (1 << kTypeShiftBits) - 1;
inline int32_t CombineTypesForTypeCast(const mindspore::ops::OP_DTYPE &src, const mindspore::ops::OP_DTYPE &dst) {
  return (static_cast<int32_t>(src) << kTypeShiftBits) | static_cast<int32_t>(dst);
}

// using OpDefConvertFunc = std::function<ValuePtr(const py::object &obj)>;
typedef ValuePtr (*OpDefConvertFunc)(const py::object &);
FRONTEND_EXPORT OpDefConvertFunc GetConverterByType(int32_t dtype);
FRONTEND_EXPORT ValuePtr ConvertTensor(const py::object &obj);
FRONTEND_EXPORT ValuePtr ConvertPyObjectTensor(PyObject *obj);

template <typename TS, typename TD, OpDefConvertFunc func>
ValuePtr ConvertSequence(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }
  auto seq = obj.cast<TS>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < seq.size(); ++it) {
    auto out = func(seq[it]);
    if (out == nullptr) {
      return nullptr;
    }
    value_list.emplace_back(out);
  }
  return std::make_shared<TD>(value_list);
}
FRONTEND_EXPORT tensor::TensorPtr ConvertTensorValue(const py::object &obj);
FRONTEND_EXPORT tensor::TensorPtr ConvertPyObjectTensorValue(PyObject *obj);

FRONTEND_EXPORT ValuePtr ConvertSlice(const py::object &obj);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PARSE_DATA_CONVERTER_H_
