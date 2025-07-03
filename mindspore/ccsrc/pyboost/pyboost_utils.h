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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include "include/common/utils/tensor_future.h"
#include "runtime/pynative/op_executor.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "runtime/device/device_address_utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_kernel_extra_func.h"
#include "utils/simple_info.h"
#include "include/common/pynative/abstract_converter.h"
#include "ops/infer_info/value_infer_info_adapter.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using AbstractConverter = pynative::AbstractConverter;
using AddressInfoPair = std::pair<std::vector<kernel::KernelTensor *>, std::vector<kernel::KernelTensorPtr>>;
using Tensor = tensor::Tensor;

class PYBOOST_API PyBoostUtils {
 public:
  static AbstractBasePtr InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs);

  static void DispatchRun(const std::shared_ptr<runtime::PyBoostDeviceTask> &task);

  static DeviceSyncPtr ContiguousByDeviceAddress(const DeviceSyncPtr &device_sync);

  // Create kernel tensors
  static std::vector<kernel::KernelTensorPtr> CreateWorkSpaceKernelTensors(const KernelModPtr &kernel_mod,
                                                                           const device::DeviceContext *device_context,
                                                                           const std::string &op_name);

  // Create output tensors
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const DeviceContext *device_context, const tensor::TensorPtr &input,
                                 const TensorStorageInfoPtr &storage_info, std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const DeviceContext *device_context, const tensor::TensorPtr &input,
                                 const TensorStorageInfoPtrList &storage_info_list,
                                 std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const ValueSimpleInfoPtr &output_value_simple_info,
                                 std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const TypeId &type_id, const ShapeVector &shape_vector,
                                 std::vector<tensor::TensorPtr> *outputs);

  // Create input device address without kernel tensor
  template <typename... Args>
  static void PrepareOpInputs(const DeviceContext *device_context, size_t stream_id, const Args &... args) {
    size_t index = 0;
    auto add_index = [&index]() { return index++; };
    (runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, stream_id, add_index(), args), ...);
  }

  template <typename... T>
  static void MallocOpInputs(const DeviceContext *device_context, const T &... args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocInput,
                                       runtime::ProfilerRecorder::kNoName, false);
    (runtime::DeviceAddressUtils::MallocForInput(device_context, args, false), ...);
  }

  static void MallocInternalOpInputs(const DeviceContext *device_context,
                                     const std::vector<tensor::TensorPtr> &tensors) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocInput,
                                       runtime::ProfilerRecorder::kNoName, false);
    for (const auto &tensor : tensors) {
      if (tensor != nullptr) {
        runtime::DeviceAddressUtils::MallocForInput(device_context, tensor, false);
      }
    }
  }

  template <typename... T>
  static void MallocOpInputsForView(const DeviceContext *device_context, const T &... args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocInput,
                                       runtime::ProfilerRecorder::kNoName, false);
    (runtime::DeviceAddressUtils::MallocForInput(device_context, args, true), ...);
  }

  template <typename... T, std::size_t... Index>
  static void GetAddressInfoHelper(const DeviceContext *device_context, size_t stream_id,
                                   const std::vector<AbstractBasePtr> &input_abs,
                                   std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list,
                                   std::index_sequence<Index...>, const T &... args) {
    (GetKernelTensor(device_context, stream_id, input_abs[Index], Index, kernel_tensor_list, kernel_tensor_ptr_list,
                     args),
     ...);
  }

  template <typename... T>
  static AddressInfoPair GetAddressInfo(const DeviceContext *device_context, size_t stream_id,
                                        const std::vector<AbstractBasePtr> &input_abs, const T &... args) {
    std::vector<kernel::KernelTensor *> kernel_tensor_list;
    // Kernel tensor is a raw ppointer, kernel tensor ptr need to be returned.
    std::vector<kernel::KernelTensorPtr> kernel_tensor_ptr_list;
    if (input_abs.empty()) {
      std::vector<AbstractBasePtr> tmp_abs(sizeof...(args), nullptr);
      GetAddressInfoHelper(device_context, stream_id, tmp_abs, &kernel_tensor_list, &kernel_tensor_ptr_list,
                           std::make_index_sequence<sizeof...(T)>(), args...);
    } else {
      GetAddressInfoHelper(device_context, stream_id, input_abs, &kernel_tensor_list, &kernel_tensor_ptr_list,
                           std::make_index_sequence<sizeof...(T)>(), args...);
    }
    return std::make_pair(kernel_tensor_list, kernel_tensor_ptr_list);
  }

  static void LaunchKernel(const PrimitivePtr &primitive, const device::DeviceContext *device_context,
                           const AddressInfoPair &input_address_info, const AddressInfoPair &output_address_info,
                           size_t stream_id = kDefaultStreamIndex, bool with_prim_attr = false);

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list, const TensorPtr &tensor) {
    GetKernelTensor(device_context, stream_id, nullptr, index, kernel_tensor_list, kernel_tensor_ptr_list, tensor);
  }

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list, const TensorPtr &tensor);

  template <typename T>
  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list,
                              const std::optional<T> &val) {
    if (val.has_value()) {
      GetKernelTensor(device_context, stream_id, input_abs, index, kernel_tensor_list, kernel_tensor_ptr_list,
                      val.value());
    } else {
      // Construct none kernel tensor
      MS_EXCEPTION_IF_NULL(kernel_tensor_list);
      MS_EXCEPTION_IF_NULL(kernel_tensor_ptr_list);

      const auto &kernel_tensor = AnfAlgo::CreateKernelTensor(
        std::make_shared<abstract::TensorShape>(ShapeVector()), kTypeNone, kNone, nullptr, 0, kOpFormat_DEFAULT,
        kTypeNone->type_id(), ShapeVector(), device_context->device_context_key().device_name_,
        device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(stream_id);
      (void)kernel_tensor_list->emplace_back(kernel_tensor.get());
      (void)kernel_tensor_ptr_list->emplace_back(kernel_tensor);
    }
  }

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list,
                              const std::vector<tensor::TensorPtr> &tensors);

  template <typename T>
  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list, const T &val) {
    // Value ptr alloc device address and malloc mem here
    auto kernel_tensor =
      runtime::DeviceAddressUtils::CreateInputKernelTensor(device_context, stream_id, input_abs, index, val);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor_ptr_list);
    MS_EXCEPTION_IF_NULL(kernel_tensor_list);
    (void)kernel_tensor_ptr_list->emplace_back(kernel_tensor);
    (void)kernel_tensor_list->emplace_back(kernel_tensor.get());
  }

  // Create output tensor device address without kernel tensor
  static void PrepareOpOutputs(const DeviceContext *device_context, size_t stream_id,
                               const std::vector<tensor::TensorPtr> &outputs) {
    runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, stream_id, outputs);
  }

  // Create output tensor device address without kernel tensor
  static void MallocOpOutputs(const DeviceContext *device_context, const std::vector<tensor::TensorPtr> &outputs) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocOutput,
                                       runtime::ProfilerRecorder::kNoName, false);
    runtime::DeviceAddressUtils::MallocForOutputs(device_context, outputs);
  }

  static std::vector<kernel::KernelTensor *> GetRawKernelTensor(
    const std::vector<kernel::KernelTensorPtr> &input_kernel_tensor);

  // Check kernel mod is reg
  static bool IsKernelModRegistered(const std::string &device_name, const std::string &op_name);
  static bool IsPyBoostCustomRegistered(const std::string &device_name, const std::string &op_name);

  // Check if enable internal kernel
  static bool IsEnableInternalKernel(const std::string &name) {
    static bool is_set;
    static bool is_enable;
    static std::set<std::string> disable_kernel_list;

    if (is_set) {
      if (!is_enable) {
        return false;
      }
      bool disable_internal_kernel =
        std::find(disable_kernel_list.begin(), disable_kernel_list.end(), name) != disable_kernel_list.end();
      return !disable_internal_kernel;
    }

    auto ms_context = MsContext::GetInstance();
    auto enable_infer_boost = ms_context->IsEnableInferBoost();
    auto enable_internal_kernel = common::GetEnv("MS_ENABLE_INTERNAL_KERNELS");
    is_enable = enable_infer_boost && (enable_internal_kernel != "off");

    std::string env = common::GetEnv("MS_DISABLE_INTERNAL_KERNELS_LIST");
    if (!env.empty()) {
      common::SplitString(env, ',', &disable_kernel_list);
    }
    bool disable_internal_kernel =
      std::find(disable_kernel_list.begin(), disable_kernel_list.end(), name) != disable_kernel_list.end();

    is_set = true;
    return is_enable && !disable_internal_kernel;
  }

  static kernel::KernelModPtr CreateKernelMod(const PrimitivePtr &prim, const DeviceContext *device_context,
                                              const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs, bool with_prim_attr);
  // return IsStrictlyMatched and KernelAttr
  static std::pair<bool, KernelAttr> SelectKernel(const std::vector<AbstractBasePtr> &inputs_abs,
                                                  const AbstractBasePtr &outputs_abs,
                                                  const DeviceContext *device_context, const std::string &op_name);
  static std::optional<tensor::TensorPtr> CastTensor(const std::optional<tensor::TensorPtr> &tensor,
                                                     const TypeId &type_id, const std::string &device_target);
  static tensor::TensorPtr CastTensor(const tensor::TensorPtr &tensor, const TypeId &type_id,
                                      const std::string &device_target);
  static std::vector<tensor::TensorPtr> CastTensor(const std::vector<tensor::TensorPtr> &tensors,
                                                   const std::vector<TypeId> &type_id_list,
                                                   const std::string &device_target);
  // ValueTuple input
  static std::vector<tensor::TensorPtr> CastTensor(const std::vector<tensor::TensorPtr> &tensors, TypeId type_id,
                                                   const std::string &device_target);
  template <typename... T>
  static std::pair<bool, KernelAttr> SelectKernel(AbstractConverter *converter, const DeviceContext *device_context,
                                                  const std::string &op_name,
                                                  const ValueSimpleInfoPtr &output_value_simple_info,
                                                  const T &... args) {
    // Get inputs abstract
    std::vector<AbstractBasePtr> input_abs;
    ((void)input_abs.emplace_back(converter->ConvertAbstract(args)), ...);

    // Get output abstract
    auto output_abs = TransformValueSimpleInfoToAbstract(*output_value_simple_info);
    return SelectKernel(input_abs, output_abs, device_context, op_name);
  }
  static ValueTuplePtr ConvertTensorVectorToTuple(const std::vector<TensorPtr> &tensor_list) {
    std::vector<ValuePtr> value_vector;
    for (const auto &tensor : tensor_list) {
      (void)value_vector.emplace_back(tensor);
    }
    auto result = std::make_shared<ValueTuple>(value_vector);
    MS_LOG(DEBUG) << "Convert TensorList to ValueTuple " << result->ToString();
    return result;
  }
  static TensorPtr ScalarToTensor(const ScalarPtr &scalar);
  static TensorPtr ScalarToTensor(const ScalarPtr &scalar, const TypePtr &tensor_dtype);

  static uint32_t cur_stream_id() { return cur_stream_id_; }

  // Set current stream for CREATE_PYBOOST_OP in front queue.
  static void set_cur_stream_id(uint32_t cur_stream_id) { cur_stream_id_ = cur_stream_id; }

  static bool IsBool(const TensorPtr &input_tensor) { return input_tensor->data_type() == TypeId::kNumberTypeBool; }

  static bool IsBool(const ScalarPtr &alpha) { return alpha->isa<BoolImm>(); }

  static bool IsIntegral(const TensorPtr &input_tensor) {
    return input_tensor->data_type() >= TypeId::kNumberTypeInt &&
           input_tensor->data_type() <= TypeId::kNumberTypeUInt64;
  }

  static bool IsIntegral(const ScalarPtr &alpha) { return alpha->isa<IntegerImm>(); }

  static bool IsFloat(const TensorPtr &input_tensor) {
    return input_tensor->data_type() >= TypeId::kNumberTypeFloat &&
           input_tensor->data_type() <= TypeId::kNumberTypeHiFloat8;
  }

  static bool IsFloat(const ScalarPtr &alpha) { return alpha->isa<FloatImm>(); }

  static bool IsComplex(const TensorPtr &input_tensor) {
    return input_tensor->data_type() >= TypeId::kNumberTypeComplex &&
           input_tensor->data_type() <= TypeId::kNumberTypeComplex128;
  }
  static void GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                  const std::string &device_target, bool is_dynamic_shape,
                                  mindspore::HashSet<size_t> *input_to_attr_index);

 private:
  inline static uint32_t cur_stream_id_ = kDefaultStreamIndex;
};

template <typename T>
std::vector<T> ConvertValueTupleToVector(const ValueTuplePtr &tuple) {
  std::vector<T> result;
  const auto &values = tuple->value();
  for (const auto &value : values) {
    (void)result.emplace_back(GetValue<T>(value));
  }
  MS_LOG(DEBUG) << "Convert ValueTuple to vector " << result;
  return result;
}

template <typename T>
std::vector<T> ConvertValueTupleToVector(const std::optional<ValueTuplePtr> &tuple) {
  if (!tuple.has_value()) {
    return {};
  }
  return ConvertValueTupleToVector<T>(tuple.value());
}

// Shield kernel hardware differences. Call some func of derived classes based on base classes.
// Just like SetThreadPool
class PYBOOST_API PyboostKernelExtraFuncFactory {
 public:
  static PyboostKernelExtraFuncFactory &GetInstance();
  PyboostKernelExtraFuncFactory() = default;
  ~PyboostKernelExtraFuncFactory() = default;
  void AddPyboostKernelExtraFunc(const std::string &op_name, const PyboostKernelExtraFuncPtr &func) {
    kernel_func_map_[op_name] = func;
  }

  void SetThreadPool(const std::string &device_name, const kernel::KernelModPtr &kernel) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return;
    }
    iter->second->SetThreadPool(kernel);
  }

  bool IsKernelModRegistered(const std::string &device_name, const std::string &op_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return true;
    }
    return iter->second->IsKernelModRegistered(op_name);
  }

  bool IsPyBoostCustomRegistered(const std::string &device_name, const std::string &op_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return true;
    }
    return iter->second->IsPyBoostCustomRegistered(op_name);
  }

  bool IsEnableProfiler(const std::string &device_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return false;
    }
    return iter->second->IsEnableProfiler();
  }

  void LaunchKernelWithProfiler(const std::string &device_name, const device::DeviceContext *device_context,
                                const std::string &op_name, const std::vector<BaseShapePtr> &base_shape,
                                const std::function<void()> &func) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return;
    }
    iter->second->LaunchKernelWithProfiler(op_name, device_context, base_shape, func);
  }

 private:
  mindspore::HashMap<std::string, PyboostKernelExtraFuncPtr> kernel_func_map_;
};

class PyboostKernelExtraFuncRegistrar {
 public:
  PyboostKernelExtraFuncRegistrar(const std::string &op_name, const PyboostKernelExtraFuncPtr &func) {
    PyboostKernelExtraFuncFactory::GetInstance().AddPyboostKernelExtraFunc(op_name, func);
  }

  ~PyboostKernelExtraFuncRegistrar() = default;
};

#define REG_PYBOOST_KERNEL_EXTRA_FUN(op_name, func) \
  static PyboostKernelExtraFuncRegistrar g_##op_name##PyboostKernelExtraFunc(#op_name, std::make_shared<func>());

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_
