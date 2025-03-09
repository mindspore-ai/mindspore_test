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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <tuple>
#include "pynative/base.h"
#include "pynative/pynative_execute.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "pynative/forward/forward_task.h"
#include "include/common/utils/tensor_py.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/visible.h"

namespace mindspore {
namespace pynative {
class PyNativeExecutor;
namespace PyNativeAlgo {
// Common function
struct Common {
  static AnfNodePtr ConvertValueSequenceToMakeTuple(const ValueNodePtr &node, const FuncGraphPtr &func_graph);
  static std::string GetIdByValue(const ValuePtr &v);
  static std::string GetCellId(const std::string &obj_id, const std::vector<std::string> &input_arg_id_vec,
                               const std::vector<ValuePtr> &input_arg_value_vec);
  static void SplitString(const std::string &str, std::vector<std::string> *id_vec);
  static bool ValueHasDynamicShape(const ValuePtr &value);
  static bool IsTensor(const ValuePtr &v, bool include_sequence = false);
  static bool IsControlFlowGraph(const FuncGraphPtr &func_graph);
  static ValuePtr FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple);
  static tensor::BaseTensorPtr GetTensorFromParam(const AnfNodePtr &param_node);
  static TypeId GetTypeFromAbstract(const abstract::AbstractBasePtr &abs);
  static ShapeVector GetShapeFromAbstract(const abstract::AbstractBasePtr &abs);
  static std::pair<TypePtr, TypeId> GetTypeFromValue(const ValuePtr &v);
  static ShapeVector GetShapeFromValue(const ValuePtr &v);
  static ValuePtr CreatOutputTensorValueByAbstract(const abstract::AbstractBasePtr &abs);
  static void ReplaceCNodeWithValueNode(const FuncGraphPtr &bprop_graph);
  static const std::shared_ptr<PyNativeExecutor> &GetPyNativeExecutor();
  static ValuePtr StubNodeToValue(const ValuePtr &val);
  static void StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info);
  static tensor::BaseTensorPtr StubNodeToTensor(const ValuePtr &value);
  PYNATIVE_EXPORT static tensor::BaseTensorPtr ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous,
                                                                       bool requires_grad);
  PYNATIVE_EXPORT static std::optional<tensor::BaseTensorPtr> ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                                      bool need_contiguous,
                                                                                      bool requires_grad);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad);
  static ValueTuplePtr ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad);
  static std::optional<ValueTuplePtr> ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                  bool need_contiguous, bool requires_grad);
  static ValuePtr StubNodeToValueInner(const ValuePtr &v);
  static void GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                  const std::string &device_target, bool is_dynamic_shape,
                                  mindspore::HashSet<size_t> *input_to_attr_index);
  static ValueNodePtr CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs = nullptr);
  static void SetOutputUsedInBpropGraph(const ValuePtr &value);
  static ValuePtr CreateFakeValueWithoutDeviceAddress(const ValuePtr &value, bool is_force_create_fake = false);
  static tensor::TensorPtr CreateFakeTensorWithoutDeviceAddress(const tensor::TensorPtr &tensor);
  static void ClearDeviceAddress(const ValuePtr &value);
  static inline bool IsConstant(InputType grad_type) { return grad_type == InputType::kConstant; }
  static void SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph);
  static void FreeFuncGraphForwardNodes(const FuncGraphPtr &func_graph);
  static tensor::BaseTensorPtr ConvertToContiguousTensor(const tensor::BaseTensorPtr &tensor, bool requires_grad);
  static ValuePtr ConvertToContiguousValue(const ValuePtr &v, bool requires_grad);
  static ValuePtr CreateTensorByConstantValue(const ValuePtr &value);

  template <typename T>
  static std::string PrintDebugInfo(std::vector<T> items, const std::string &info_header = "",
                                    bool is_print_tensor_data = false) {
    static constexpr size_t end_char_size = 2;
    std::ostringstream buf;
    buf << info_header;
    for (size_t i = 0; i < items.size(); ++i) {
      if (items[i] == nullptr) {
        MS_LOG(DEBUG) << "The " << i << "'th item is nullptr!";
        continue;
      }
      if (items[i]->template isa<tensor::BaseTensor>() && is_print_tensor_data) {
        auto tensor = items[i]->template cast<tensor::BaseTensorPtr>();
        auto grad = std::make_shared<tensor::Tensor>(*tensor);
        grad->data_sync();
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << grad->ToStringRepr() << ", ";
      } else {
        buf << i << "th: "
            << "ptr " << items[i].get() << ", " << items[i]->ToString() << ", ";
      }
    }
    return buf.str().erase(buf.str().size() - end_char_size);
  }
  static bool IsHookNeedSaveInputs(const PrimitivePyPtr &prim);
  static bool IsVmOp(const std::string &op_name);
  static std::vector<int64_t> BuildShape(const abstract::AbstractBasePtr &abs);
  static void ClearRes();
  static TopCellInfo *FindPreTopcell(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                                     const std::string &op_info, const ValuePtr &value);
  static void UpdateGradOpInfo(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                               TopCellInfo *pre_top_cell, bool is_jit_graph);
  static OperatorType GetOpTypeFromOpdef(const ops::OpDef &op_def);
};

// Parser python
struct PyParser {
  static std::string GetIdByPyObj(const py::object &obj);
  static std::pair<std::vector<std::string>, std::vector<ValuePtr>> GetArgsIdAndValue(const py::args &args);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs,
                                      bool stub = false);
  static std::string BuilidPyInputTypeString(const py::object &obj);

  static inline bool IsSupportTensorCast(const std::vector<ops::OP_DTYPE> &cast_types) {
    for (const auto &type : cast_types) {
      if (type == ops::DT_TENSOR) {
        return true;
      }
    }
    return false;
  }
  static void PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx);
};

// Data convert
struct DataConvert {
  static ValuePtrList FlattenOnlyTensor(const ValuePtr &v);
  static void FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v, bool has_sens);
  static ValuePtrList FlattenTensorSeqInValue(const ValuePtr &v);
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, const TopCellInfoPtr &top_cell,
                                           size_t index);
  static void ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                               const TopCellInfoPtr &top_cell, size_t index);
  static ValuePtr ConvertValueDictToValueTuple(const ValuePtr &v);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       size_t index, const TopCellInfoPtr &top_cell);
  static void GetTensorIdFromOutputValue(const ValuePtr &value, std::vector<std::string> *converted_tensor_id);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        size_t index, const TopCellInfoPtr &top_cell);
  static void MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                         const TopCellInfoPtr &top_cell);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index);
  static void TransformValueNodeBaseTensorToTensor(const ValueNodePtr &value_node);
  static ValuePtr ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs);
  static ValuePtrList TensorListToValueList(const tensor::BaseTensorPtrList &tensor_list);
};

struct PyBoost {
  static FrontendOpRunInfoPtr Init(const PrimitivePtr &prim);
  static void DoGrad(const kernel::pyboost::OpPtr &op, const OpGradInfoPtr &grad_info, const AsyncStatus &async_status);
  static void SetAnyValueForAbstract(const kernel::pyboost::OpPtr &op);
  static void UpdateStubOutput(const kernel::pyboost::OpPtr &op, const stub::StubNodePtr &stub_output,
                               const AbstractBasePtr &abstract, const ValuePtr &real_out);
  static PrimitivePtr ConvertPrimitive(const py::object &obj);
  static py::object RunPyFunction(const PrimitivePtr &prim, const py::list &args);
  template <typename T>
  static ValuePtr OptionalToValue(const std::optional<T> &val) {
    if (!val.has_value()) {
      return kNone;
    }
    return val.value();
  }

  template <typename Tuple, size_t... N>
  static std::vector<ValuePtr> TupleToVector(const Tuple &tuple, std::index_sequence<N...>) {
    std::vector<ValuePtr> inputs;
    ((void)inputs.emplace_back(OptionalToValue(std::get<N>(tuple))), ...);
    return inputs;
  }

  template <typename T>
  static T OptionalToValue(const T &val) {
    return val;
  }

  template <size_t N, typename... T>
  static auto SetPyBoostCastForInputs(const FrontendOpRunInfoPtr &op_run_info,
                                      const std::vector<std::vector<size_t>> &same_type_table, T... t) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    op_run_info->input_size = sizeof...(t);
    if (op_run_info->op_grad_info->op_prim->name() == kCast) {
      return std::make_tuple(t...);
    }
    const auto &pyboost_cast_operation = Common::GetPyNativeExecutor()->forward_executor()->pyboost_cast_operation();
    const auto &ret = pyboost_cast_operation->DoMixPrecisionCast(op_run_info, t...);
    if constexpr (N != 0) {
      return pyboost_cast_operation->DoImplicitCast<N>(op_run_info, same_type_table, ret);
    }
    return ret;
  }
  static void DataSyncForGraph(const kernel::pyboost::OpPtr &op);
  static void MarkPyBoostInputs(const OpGradInfoPtr &op_grad_info, const TopCellInfoPtr &top_cell);
  static void BumpVersionAsync(const tensor::BaseTensorPtr &tensor);
  static ValuePtr OutputToValue(const BaseTensorPtr &output) { return output; }
  static ValuePtr MultiOutputToValue(const std::vector<BaseTensorPtr> &outputs) {
    std::vector<ValuePtr> output_values;
    output_values.reserve(outputs.size());
    (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_values),
                         [](const BaseTensorPtr &value) -> ValuePtr { return value; });
    return std::make_shared<ValueTuple>(output_values);
  }
  template <typename... Args>
  static ValuePtr MultiOutputToValue(const std::tuple<Args...> &outputs) {
    std::vector<ValuePtr> output_values = TupleToVector(outputs);
    return std::make_shared<ValueTuple>(output_values);
  }

 private:
  template <std::size_t... Is, typename Tuple>
  static std::vector<ValuePtr> UnpackTuple(const Tuple &t, std::index_sequence<Is...>) {
    return {std::get<Is>(t)...};
  }

  template <typename... Args>
  static std::vector<ValuePtr> TupleToVector(const std::tuple<Args...> &t) {
    return UnpackTuple(t, std::index_sequence_for<Args...>{});
  }
};

// Some common functions used in both jit and PackFunc grad
struct GradCommon {
  static bool IsRealOp(const AnfNodePtr &cnode);
  static bool HasTensorOutput(const abstract::AbstractBasePtr &abs);
  static void GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                                       AnfNodePtrList *node_list);
  static void SetForward(const AnfNodePtrList &node_list);
};
};  // namespace PyNativeAlgo

PYNATIVE_EXPORT void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
