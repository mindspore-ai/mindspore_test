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

#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include <algorithm>
#include <utility>
#include <unordered_map>
#include "common/common_utils.h"
#include "common/kernel_mod_cache.h"
#include "mindapi/base/type_id.h"
#include "runtime/device/device_address_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/infer_info/infer_info_utils.h"
#include "ops/op_def.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pipeline/pipeline.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/ccsrc/pyboost/auto_generate/cast.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ccsrc/include/backend/optimizer/helper.h"
#include "mindspore/ccsrc/include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void CreateTensor(const TypeId &type_id, const ShapeVector &shape_vector, const AbstractBasePtr &abstract_tensor,
                  std::vector<tensor::TensorPtr> *outputs) {
  auto output_tensor = std::make_shared<tensor::Tensor>(type_id, shape_vector);
  output_tensor->set_need_pipeline_sync(true);
  (void)outputs->emplace_back(output_tensor);
  MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
}

void CreateTensor(const TypeId &type_id, const ShapeVector &shape_vector, std::vector<tensor::TensorPtr> *outputs) {
  auto output_tensor = std::make_shared<tensor::Tensor>(type_id, shape_vector);
  output_tensor->set_need_pipeline_sync(true);
  (void)outputs->emplace_back(output_tensor);
  MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
}
}  // namespace

void PyBoostUtils::CreateOutputTensor(const TypeId &type_id, const ShapeVector &shape_vector,
                                      std::vector<tensor::TensorPtr> *outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyBoostCreateOutputTensor,
                                     runtime::ProfilerRecorder::kNoName, false);
  CreateTensor(type_id, shape_vector, outputs);
}

void PyBoostUtils::CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyBoostCreateOutputTensor,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &seq = abstract->cast<abstract::AbstractSequencePtr>();
    const auto &elements = seq->elements();
    for (const auto &element : elements) {
      CreateOutputTensor(element, outputs);
    }
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    const auto &shape = abstract_tensor->GetShapeTrack();
    const auto &type = abstract_tensor->element()->GetTypeTrack();
    MS_LOG(DEBUG) << "get abstract tensor shape " << shape->ToString() << " type " << type->ToString();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "AbstractTensor shape is valid " << shape->ToString();
    }
    const auto &shape_vector = shape->cast<abstract::ShapePtr>()->shape();
    CreateTensor(type->type_id(), shape_vector, abstract_tensor, outputs);
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    const auto &scalar = abstract->cast<abstract::AbstractScalarPtr>();
    const auto &type = scalar->GetTypeTrack();
    MS_LOG(DEBUG) << "Create scalar tensor type " << type->ToString();
    CreateTensor(type->type_id(), {}, nullptr, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Not support abstract " << abstract->ToString();
  }
}

tensor::TensorPtr PyBoostUtils::ScalarToTensor(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  return ScalarToTensor(scalar, scalar->type());
}

tensor::TensorPtr PyBoostUtils::ScalarToTensor(const ScalarPtr &scalar, const TypePtr &tensor_dtype) {
  MS_EXCEPTION_IF_NULL(scalar);
  MS_EXCEPTION_IF_NULL(tensor_dtype);

  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return std::make_shared<tensor::Tensor>(GetValue<bool>(scalar), tensor_dtype);
    case kNumberTypeInt8:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int8_t>(scalar)), tensor_dtype);
    case kNumberTypeInt16:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int16_t>(scalar)), tensor_dtype);
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int32_t>(scalar)), tensor_dtype);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(GetValue<int64_t>(scalar), tensor_dtype);
    case kNumberTypeUInt8:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint8_t>(scalar)), tensor_dtype);
    case kNumberTypeUInt16:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint16_t>(scalar)), tensor_dtype);
    case kNumberTypeUInt32:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint32_t>(scalar)), tensor_dtype);
    case kNumberTypeUInt64:
      return std::make_shared<tensor::Tensor>(GetValue<uint64_t>(scalar), tensor_dtype);
    case kNumberTypeFloat32:
      return std::make_shared<tensor::Tensor>(GetValue<float>(scalar), tensor_dtype);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(GetValue<double>(scalar), tensor_dtype);
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the scalar type: " << data_type << " is invalid.";
  }
}

void PyBoostUtils::CreateOutputTensor(const ValueSimpleInfoPtr &output_value_simple_info,
                                      std::vector<tensor::TensorPtr> *outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyBoostCreateOutputTensor,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_EXCEPTION_IF_NULL(output_value_simple_info);
  size_t elem_size = output_value_simple_info->dtype_vector_.size();
  for (size_t i = 0; i < elem_size; ++i) {
    MS_LOG(DEBUG) << "Get tensor shape " << output_value_simple_info->shape_vector_[i] << ", type "
                  << TypeIdToType(output_value_simple_info->dtype_vector_[i]->type_id())->ToString();
    CreateTensor(output_value_simple_info->dtype_vector_[i]->type_id(), output_value_simple_info->shape_vector_[i],
                 outputs);
  }
}

bool PyBoostUtils::IsKernelModRegistered(const std::string &device_name, const std::string &op_name) {
  return PyboostKernelExtraFuncFactory::GetInstance().IsKernelModRegistered(device_name, op_name);
}

bool PyBoostUtils::IsPyBoostCustomRegistered(const std::string &device_name, const std::string &op_name) {
  return PyboostKernelExtraFuncFactory::GetInstance().IsPyBoostCustomRegistered(device_name, op_name);
}

kernel::KernelModPtr PyBoostUtils::CreateKernelMod(const PrimitivePtr &prim, const DeviceContext *device_context,
                                                   const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs, bool with_prim_attr) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &device_name = device_context->device_context_key().device_name_;
  const auto &op_name = prim->name();

  auto &cache_helper = kernel::KernelModCache::GetInstance();

  const auto &key = with_prim_attr ? cache_helper.GetPrimAttrKernelModKey(prim, device_name, inputs)
                                   : cache_helper.GetKernelModKey(op_name, device_name, inputs);
  auto kernel_mod = cache_helper.GetKernelMod(key);
  if (kernel_mod == nullptr) {
    kernel_mod = device_context->GetKernelExecutor()->CreateKernelMod(op_name);
    if (kernel_mod == nullptr) {
      if (common::EnvHelper::GetInstance()->GetEnv("MS_OP_PLUGIN_PATH") != nullptr) {
        // if env var MS_OP_PLUGIN_PATH is set, then use custom op plugin to load op
        const std::string custom_op_name = "CustomOpPlugin";
        kernel_mod = device_context->GetKernelExecutor()->CreateKernelMod(custom_op_name);
        MS_EXCEPTION_IF_NULL(kernel_mod);
      } else {
        MS_LOG(EXCEPTION) << "Create kernelmod for op " << op_name << " failed";
      }
    }
    if (!kernel_mod->Init(prim, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "KernelMod Init Failed: " << op_name;
    }
    cache_helper.SetCache(key, kernel_mod);
    PyboostKernelExtraFuncFactory::GetInstance().SetThreadPool(device_name, kernel_mod);
  }

  return kernel_mod;
}

DeviceSyncPtr PyBoostUtils::ContiguousByDeviceAddress(const DeviceSyncPtr &device_sync) {
  auto &storage_info = device_sync->GetTensorStorageInfo();
  if (storage_info == nullptr) {
    return device_sync;
  }

  auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);

  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(storage_info);
  GilReleaseWithCheck gil_release;

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {old_device_address->device_name(), old_device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);

  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  auto address_size = GetTypeByte(TypeIdToType(old_device_address->type_id())) * SizeOf(storage_info->shape);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, address_size, storage_info->shape, DEFAULT_FORMAT, old_device_address->type_id(),
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_, stream_id);
  new_device_address->set_device_shape(storage_info->shape);
  new_device_address->set_new_ref_count(SIZE_MAX);

  if (!device_context->GetKernelExecutor()->ExecuteKernelTask(runtime::KernelTaskType::kCONTIGUOUS_TASK,
                                                              {old_device_address}, {new_device_address}, stream_id)) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << runtime::KernelTaskType::kCONTIGUOUS_TASK;
  }
  runtime::Pipeline::Get().WaitForward();
  return new_device_address;
}

void PyBoostUtils::CreateOutputTensor(const DeviceContext *device_context, const tensor::TensorPtr &input,
                                      const TensorStorageInfoPtrList &storage_info_list,
                                      std::vector<tensor::TensorPtr> *outputs) {
  for (auto &storage_info : storage_info_list) {
    CreateOutputTensor(device_context, input, storage_info, outputs);
  }
}

void PyBoostUtils::CreateOutputTensor(const DeviceContext *device_context, const tensor::TensorPtr &input,
                                      const TensorStorageInfoPtr &storage_info,
                                      std::vector<tensor::TensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(storage_info);
  MS_EXCEPTION_IF_NULL(device_context);

  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyBoostCreateOutputTensor,
                                     runtime::ProfilerRecorder::kNoName, false);
  auto output_tensor = std::make_shared<tensor::Tensor>(input->data_type(), storage_info->shape);
  output_tensor->set_need_pipeline_sync(true);
  output_tensor->set_contiguous_callback(
    [](const DeviceSyncPtr &device_address) -> DeviceSyncPtr { return ContiguousByDeviceAddress(device_address); });

  auto input_device_address = std::static_pointer_cast<device::DeviceAddress>(input->device_address());
  MS_EXCEPTION_IF_NULL(input_device_address);
  input_device_address->set_is_view(true);

  // Create view output address
  auto output_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, input_device_address->GetSize(), output_tensor->shape(), DEFAULT_FORMAT, output_tensor->data_type(),
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_,
    input_device_address->stream_id());
  MS_EXCEPTION_IF_NULL(output_device_address);
  output_device_address->set_tensor_storage_info(storage_info);
  output_device_address->set_pointer_ref_count(input_device_address->pointer_ref_count());
  output_tensor->set_device_address(output_device_address);
  (void)outputs->emplace_back(output_tensor);
  MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString() << " with " << storage_info->ToString();
}

AbstractBasePtr PyBoostUtils::InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs) {
  MS_EXCEPTION_IF_NULL(prim);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostInferByOpDef,
                                     prim->name(), false);
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def) {
    AbstractBasePtr output_abs;
    if (op_def->func_impl_.GeneralInferRegistered()) {
      output_abs = ops::DoGeneralInfer(prim, input_abs);
    } else {
      (void)op_def->func_impl_.CheckValidation(prim, input_abs);
      auto shape = op_def->func_impl_.InferShape(prim, input_abs);
      auto type = op_def->func_impl_.InferType(prim, input_abs);
      output_abs = mindspore::abstract::MakeAbstract(shape, type);
    }
    MS_LOG(DEBUG) << "Pynative Infer " << prim->name() << " by OpDef, got abstract: " << output_abs->ToString();
    return output_abs;
  } else {
    const auto &infer_map = abstract::GetPrimitiveInferMapPtr();
    const auto &iter = infer_map->find(prim);
    if (iter != infer_map->end()) {
      auto output_abs = iter->second.InferShapeAndType(nullptr, prim, input_abs);
      MS_LOG(DEBUG) << "Pynative Infer " << prim->name()
                    << " by C++ PrimitiveInferMap, got abstract: " << output_abs->ToString();
      return output_abs;
    } else {
      MS_LOG(EXCEPTION) << "Cannot found infer function for Op " << prim->name();
    }
  }
}

void PyBoostUtils::DispatchRun(const std::shared_ptr<runtime::PyBoostDeviceTask> &task) {
  static auto need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    MS_LOG(INFO) << "PyBoost sync run device task";
    runtime::Pipeline::Get().WaitAll();
    task->Run();
  } else {
    runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
    runtime::OpExecutor::GetInstance().PushOpRunTask(task);
  }
}

void PyBoostUtils::GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                   const abstract::AbstractBasePtr &input_abs, size_t index,
                                   std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list,
                                   const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor_list);
  MS_EXCEPTION_IF_NULL(kernel_tensor_ptr_list);

  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  auto tmp_abs = input_abs;
  if (input_abs == nullptr) {
    tmp_abs = tensor->ToAbstract()->Broaden();
  }
  auto shape = tmp_abs->GetShape();
  auto type = tmp_abs->GetType();
  auto value = tmp_abs->GetValue();
  auto kernel_tensor = std::make_shared<KernelTensor>(shape, type, value);
  kernel_tensor->set_device_address(device_address);
  kernel_tensor->set_host_shape(tensor->shape());
  MS_LOG(DEBUG) << "Create input " << tmp_abs->ToString() << " device address for " << index
                << "th input, Shape: " << shape->ToString() << ", Type: " << type->ToString()
                << ", Value: " << (value ? value->ToString() : "nullptr") << " device address:" << device_address.get()
                << ", kernel tensor: " << kernel_tensor.get();
  (void)kernel_tensor_ptr_list->emplace_back(kernel_tensor);
  (void)kernel_tensor_list->emplace_back(kernel_tensor.get());
}

std::vector<kernel::KernelTensor *> PyBoostUtils::GetRawKernelTensor(
  const std::vector<kernel::KernelTensorPtr> &input_kernel_tensor) {
  std::vector<kernel::KernelTensor *> input_kernel_tensors;
  std::transform(input_kernel_tensor.begin(), input_kernel_tensor.end(), std::back_inserter(input_kernel_tensors),
                 [](const auto &item) { return item.get(); });
  return input_kernel_tensors;
}

void PyBoostUtils::GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                   const abstract::AbstractBasePtr &input_abs, size_t index,
                                   std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   std::vector<kernel::KernelTensorPtr> *kernel_tensor_ptr_list,
                                   const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    // input_abs is not used in GetKernelTensor when value is TensorPtr.
    GetKernelTensor(device_context, stream_id, input_abs, index, kernel_tensor_list, kernel_tensor_ptr_list, tensor);
  }
}

std::vector<kernel::KernelTensorPtr> PyBoostUtils::CreateWorkSpaceKernelTensors(
  const KernelModPtr &kernel_mod, const device::DeviceContext *device_context, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  const auto &workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  std::vector<kernel::KernelTensorPtr> workspaces_kernel_tensors;
  for (const auto workspace_size : workspace_sizes) {
    auto kernel_tensor = AnfAlgo::CreateKernelTensor(nullptr, workspace_size, Format::DEFAULT_FORMAT, kTypeUnknown,
                                                     ShapeVector(), device_context->device_context_key().device_name_,
                                                     device_context->device_context_key().device_id_);
    (void)workspaces_kernel_tensors.emplace_back(kernel_tensor);
  }

  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto kernel_tensor = workspaces_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kWorkSpace,
                                                   device_address->GetSize(), device_address.get());
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << device_address->GetPtr() << " size:" << device_address->GetSize();
  }
  return workspaces_kernel_tensors;
}

PyboostKernelExtraFuncFactory &PyboostKernelExtraFuncFactory::GetInstance() {
  static PyboostKernelExtraFuncFactory instance;
  return instance;
}

void PyBoostUtils::LaunchKernel(const PrimitivePtr &primitive, const DeviceContext *device_context,
                                const AddressInfoPair &input_address_info, const AddressInfoPair &output_address_info,
                                size_t stream_id, bool with_prim_attr) {
  const auto &real_name = primitive->name();
  // KernelMod init
  auto kernel_mod = PyBoostUtils::CreateKernelMod(primitive, device_context, input_address_info.first,
                                                  output_address_info.first, with_prim_attr);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // KernelMod resize
  if (kernel_mod->Resize(input_address_info.first, output_address_info.first) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << real_name << "] resize failed.";
  }
  // Get workspace kernel tensors
  const auto &workspace_kernel_tensors =
    PyBoostUtils::CreateWorkSpaceKernelTensors(kernel_mod, device_context, primitive->name());

  auto workspaces = PyBoostUtils::GetRawKernelTensor(workspace_kernel_tensors);

  auto launch_kernel_func = [&]() {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask,
                                       real_name, false);
    void *stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
    if (!kernel_mod->Launch(input_address_info.first, workspaces, output_address_info.first, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << real_name;
    }
    static bool sync_stream = runtime::RuntimeConf::GetInstance()->launch_blocking();
    if (sync_stream) {
      if (!device_context->device_res_manager_->SyncStream(stream_id)) {
        MS_LOG(EXCEPTION) << "SyncStream failed, op name " << real_name;
      }
    }
  };

  const auto &device_name = device_context->device_context_key().device_name_;
  if (!PyboostKernelExtraFuncFactory::GetInstance().IsEnableProfiler(device_name)) {
    launch_kernel_func();
  } else {
    PyboostKernelExtraFuncFactory::GetInstance().LaunchKernelWithProfiler(device_name, device_context, real_name, {},
                                                                          [&]() { launch_kernel_func(); });
  }
  if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod->UpdateOutputShapeAndSize(input_address_info.first, output_address_info.first);
  }
  runtime::DeviceAddressUtils::ProcessCrossStreamAddress(real_name, device_context, stream_id, input_address_info.first,
                                                         output_address_info.first);
  MS_LOG(DEBUG) << real_name << " Launch end";
}

void PyBoostUtils::GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                       const std::string &device_target, bool is_dynamic_shape,
                                       mindspore::HashSet<size_t> *input_to_attr_index) {
  if (op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    PrimitiveReadLock read_lock(op_prim->shared_mutex());
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_index);
    return;
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return;
  }

  auto reg_info =
    opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, device_target, is_dynamic_shape);
  if (reg_info == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(input_to_attr_index);
  for (auto &iter : reg_info->input_attr_map()) {
    (void)input_to_attr_index->insert(iter.first);
  }
}

namespace {
TypeId GetTypeIdFromAbstractTensor(const AbstractBasePtr &abs_base) {
  if (abs_base->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = std::dynamic_pointer_cast<abstract::AbstractTensor>(abs_base);
    return abs_tensor->element()->BuildType()->type_id();
  }
  return abs_base->BuildType()->type_id();
}

std::pair<std::vector<TypeId>, std::vector<TypeId>> GetOutputTypeFromAbstractBase(const AbstractBasePtr &abs_base) {
  std::vector<TypeId> output_dtype;
  std::vector<TypeId> output_type;
  if (abs_base->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = std::dynamic_pointer_cast<abstract::AbstractTuple>(abs_base);
    for (auto &abs : abs_tuple->elements()) {
      (void)output_dtype.emplace_back(GetTypeIdFromAbstractTensor(abs));
      (void)output_type.emplace_back(session::AnfRuntimeAlgorithm::GetAbstractObjectType(abs));
    }
  } else {
    (void)output_type.emplace_back(session::AnfRuntimeAlgorithm::GetAbstractObjectType(abs_base));
    (void)output_dtype.emplace_back(GetTypeIdFromAbstractTensor(abs_base));
  }
  return std::make_pair(output_type, output_dtype);
}

std::pair<std::vector<TypeId>, std::vector<TypeId>> GetInputTypeFromAbstractBase(
  const std::vector<AbstractBasePtr> &abs_vec) {
  std::vector<TypeId> input_dtype;
  std::vector<TypeId> input_type;
  for (auto &abs : abs_vec) {
    if (abs->isa<abstract::AbstractTuple>()) {
      // a tuple tensors have same type
      auto abs_tuple = std::dynamic_pointer_cast<abstract::AbstractTuple>(abs);
      if (abs_tuple->elements().empty()) {
        input_dtype.emplace_back(kTypeUnknown);
        continue;
      }
      input_dtype.emplace_back(abs_tuple->elements()[0]->BuildType()->type_id());
    } else {
      input_dtype.emplace_back(GetTypeIdFromAbstractTensor(abs));
    }
    input_type.emplace_back(session::AnfRuntimeAlgorithm::GetAbstractObjectType(abs));
  }
  return std::make_pair(input_type, input_dtype);
}

bool InputDtypeMatch(TypeId input_attr, TypeId input_type) {
  if (input_attr == input_type || kTypeUnknown == input_type) {
    return true;
  }
  if (input_attr == kNumberTypeInt32 && (input_type == kNumberTypeInt16 || input_type == kNumberTypeInt64)) {
    return true;
  }
  if (input_attr == kNumberTypeFloat32 && (input_type == kNumberTypeFloat16 || input_type == kNumberTypeFloat64)) {
    return true;
  }
  return false;
}

bool IsObjectDtypeWeaklyMatched(const std::vector<TypeId> &object_dtypes,
                                const std::vector<DataType> &kernel_data_types) {
  // only support CPU
  for (size_t i = 0; i < object_dtypes.size(); i++) {
    // For optional input, the real input object type can be a None.
    if (!InputDtypeMatch(kernel_data_types[i].dtype, object_dtypes[i])) {
      return false;
    }
  }
  return true;
}

bool IsObjectStrictlyMatched(const std::vector<TypeId> &object_types, const std::vector<TypeId> &object_dtypes,
                             const std::vector<DataType> &kernel_data_types) {
  if (object_dtypes.size() != kernel_data_types.size()) {
    return false;
  }

  for (size_t i = 0; i < object_dtypes.size(); i++) {
    auto is_tuple = (kernel_data_types[i].object_type == kObjectTypeTuple);
    // For optional input, the real input object type can be a None.Tuple data-type unknown means empty tuple.
    if (object_dtypes[i] != kernel_data_types[i].dtype) {
      if ((!is_tuple || object_dtypes[i] != kTypeUnknown) &&
          !(object_types[i] == kMetaTypeNone && kernel_data_types[i].is_optional)) {
        return false;
      }
    }
  }

  return true;
}

std::pair<bool, KernelAttr> GetKernelAttr(
  const std::string &op_name, const kernel::KernelModPtr &kernel_mod,
  const std::pair<std::vector<TypeId>, std::vector<TypeId>> &inputs_types_dtypes,
  const std::pair<std::vector<TypeId>, std::vector<TypeId>> &outputs_types_dtypes) {
  const auto &support_list = kernel_mod->GetOpSupport();
  for (auto &cur_kernel_attr : support_list) {
    if (cur_kernel_attr.GetSkipCheck()) {
      return {true, cur_kernel_attr};
    }
    auto data_pair = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    const auto &[input_data_types, output_data_types] = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    if (IsObjectStrictlyMatched(inputs_types_dtypes.first, inputs_types_dtypes.second, input_data_types) &&
        IsObjectStrictlyMatched(outputs_types_dtypes.first, outputs_types_dtypes.second, output_data_types)) {
      return std::make_pair(true, cur_kernel_attr);
    }
  }

  for (auto &cur_kernel_attr : support_list) {
    auto data_pair = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    const auto &[input_data_types, output_data_types] = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    if (IsObjectDtypeWeaklyMatched(inputs_types_dtypes.second, input_data_types) &&
        IsObjectDtypeWeaklyMatched(outputs_types_dtypes.second, output_data_types)) {
      return std::make_pair(false, cur_kernel_attr);
    }
  }
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  for (auto &input_type : inputs_types_dtypes.second) {
    (void)inputs.emplace_back(TypeIdToString(input_type));
  }
  for (auto &output_type : outputs_types_dtypes.second) {
    (void)outputs.emplace_back(TypeIdToString(output_type));
  }
  std::string support_types;
  for (auto &cur_kernel_attr : support_list) {
    auto data_pair = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    const auto &[input_data_types, output_data_types] = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    std::string input_types_str = TypeIdToString(input_data_types[0].dtype);
    for (size_t i = 1; i < input_data_types.size(); ++i) {
      input_types_str += " " + TypeIdToString(input_data_types[i].dtype);
    }
    std::string input_str = "input[" + input_types_str + "]";

    std::string output_types_str = TypeIdToString(output_data_types[0].dtype);
    for (size_t i = 1; i < output_data_types.size(); ++i) {
      output_types_str += " " + TypeIdToString(output_data_types[i].dtype);
    }
    std::string output_str = "output[" + output_types_str + "]";
    support_types += input_str + ", " + output_str + ";";
  }

  MS_EXCEPTION(TypeError) << "Select CPU operator[" << op_name
                          << "] fail! Unsupported data type!\nThe supported data types are " << support_types
                          << ",\n but get input[" << inputs << "], output[" << outputs << "].";
}
}  // namespace

std::pair<bool, KernelAttr> PyBoostUtils::SelectKernel(const std::vector<AbstractBasePtr> &inputs_abs,
                                                       const AbstractBasePtr &outputs_abs,
                                                       const DeviceContext *device_context,
                                                       const std::string &op_name) {
  // only support CPU
  const auto &kernel_mod = device_context->GetKernelExecutor()->CreateKernelMod(op_name);
  if (kernel_mod == nullptr) {
    if (common::EnvHelper::GetInstance()->GetEnv("MS_OP_PLUGIN_PATH") != nullptr) {
      // if env var MS_OP_PLUGIN_PATH is set, then use custom op plugin to load op
      MS_LOG(INFO) << "The kernel " << op_name << " unregistered, use custom op plugin";
      return {true, KernelAttr()};
    }
    MS_LOG(EXCEPTION) << "The kernel " << op_name << " unregistered.";
  }
  return GetKernelAttr(op_name, kernel_mod, GetInputTypeFromAbstractBase(inputs_abs),
                       GetOutputTypeFromAbstractBase(outputs_abs));
}

std::optional<tensor::TensorPtr> PyBoostUtils::CastTensor(const std::optional<tensor::TensorPtr> &tensor,
                                                          const TypeId &type_id, const std::string &device_target) {
  if (!tensor.has_value()) {
    return tensor;
  }
  if (tensor.value()->Dtype()->type_id() == type_id) {
    return tensor;
  }
  auto type_id64 = std::make_shared<Int64Imm>(static_cast<int64_t>(type_id));
  const auto &cast_op = CREATE_PYBOOST_OP(Cast, device_target);
  cast_op->set_primitive(prim::kPrimCast);
  return cast_op->Call(tensor.value(), type_id64);
}

tensor::TensorPtr PyBoostUtils::CastTensor(const tensor::TensorPtr &tensor, const TypeId &type_id,
                                           const std::string &device_target) {
  if (tensor->Dtype()->type_id() == type_id) {
    return tensor;
  }
  auto type_id64 = std::make_shared<Int64Imm>(static_cast<int64_t>(type_id));
  const auto &cast_op = CREATE_PYBOOST_OP(Cast, device_target);
  return cast_op->Call(tensor, type_id64);
}

std::vector<tensor::TensorPtr> PyBoostUtils::CastTensor(const std::vector<tensor::TensorPtr> &tensors,
                                                        const std::vector<TypeId> &type_id_list,
                                                        const std::string &device_target) {
  if (tensors.size() != type_id_list.size()) {
    MS_LOG(EXCEPTION) << "before cast tensor output size is not equal after cast";
  }
  std::vector<tensor::TensorPtr> output_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &output = CastTensor(tensors[i], type_id_list[i], device_target);
    (void)output_tensors.emplace_back(output);
  }
  return output_tensors;
}

std::vector<tensor::TensorPtr> PyBoostUtils::CastTensor(const std::vector<tensor::TensorPtr> &tensors, TypeId type_id,
                                                        const std::string &device_target) {
  // tuple input
  std::vector<tensor::TensorPtr> output_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &output = CastTensor(tensors[i], type_id, device_target);
    (void)output_tensors.emplace_back(output);
  }
  return output_tensors;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
