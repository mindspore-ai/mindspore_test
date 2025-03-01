/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_COMMON_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_COMMON_UTILS_H_

#include <string>
#include <vector>
#include <utility>
#include "include/common/visible.h"
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "base/base.h"
#include "ir/func_graph.h"
#include "include/common/utils/hook.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class COMMON_EXPORT CommonUtils {
 public:
  static void ProcessTupleParam(const FuncGraphPtr &bprop_graph, size_t position);
  static void ProcessDictParam(const FuncGraphPtr &bprop_graph, size_t position);
  static void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  static AbstractBasePtr SetAbstractValueToAnyValue(const AbstractBasePtr &abs);
  static ValuePtrList FlattenTensorSeqInValueSeq(const ValuePtrList &v, bool only_flatten_tensor = true);
  static void FlattenValueSeqArg(const ValuePtr &v, bool is_only_flatten_tensor_seq, bool is_filter_tensor,
                                 std::vector<ValuePtr> *flatten_v);
};

class COMMON_EXPORT CastUtils {
 public:
  static ValuePtr ScalarToDstDtypeValue(const ValuePtr &src_value, const std::pair<TypeId, bool> &dst_type);
  static tensor::BaseTensorPtr TensorToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id);
};
}  // namespace pynative

class COMMON_EXPORT HookUtils {
 public:
  static bool HasRegisterHook(const py::object &obj);
  static py::list GetRegisterHookList(const py::object &obj);
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_COMMON_UTILS_H_
