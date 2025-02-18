/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/dvm/lazy_fusion_kernel.h"
#include "plugin/device/ascend/kernel/dvm/lazy_fusion_flags.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "include/backend/debug/profiler/profiling.h"
#include "runtime/pipeline/pipeline.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
namespace {
void *WsAllocCallback(uint64_t size, void *user_data) {
  auto kernel = static_cast<LazyFusionKernelAscend *>(user_data);
  return kernel->AllocWorkspace(size);
}
}  // namespace

void LazyFusionQueue::Push(const runtime::AsyncTaskPtr &task) {
  FlushLazyFusion();
  AsyncRQueue::Push(task);
}

void LazyFusionQueue::Wait() {
  FlushLazyFusion();
  AsyncRQueue::Wait();
}

LazyFusionManager g_lazy_fusion_manager;

LazyFusionManager::~LazyFusionManager() {
  while (!pool_.empty()) {
    auto top = pool_.front();
    delete top;
    pool_.pop();
  }
}

LazyFusionKernelAscend *LazyFusionManager::Get(const device::DeviceContext *context, size_t stream) {
  if (current_ != nullptr) {
    if (current_->stream_id() == stream) {
      return current_;
    }
    current_->Flush();
  }
  current_ = NewKernel();
  current_->Reset(context, stream);
  current_->set_id(id_.fetch_add(1, std::memory_order_relaxed));
  return current_;
}

void LazyFusionManager::Flush() {
  if (current_ != nullptr) {
    current_->Flush();
    current_ = nullptr;
  }
}

LazyFusionKernelAscend *LazyFusionManager::NewKernel() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pool_.empty()) {
      auto k = pool_.front();
      pool_.pop();
      return k;
    }
  }
  return new LazyFusionKernelAscend();
}

LazyFusionKernelAscend::LazyFusionKernelAscend() { EagerReset(WsAllocCallback, this); }

LazyFusionKernelAscend::~LazyFusionKernelAscend() {
  for (auto load : inputs_) {
    delete load;
  }
}

dvm::ShapeRef *LazyFusionKernelAscend::GetShapeRef(const ShapeVector &shape) {
  auto &item = cached_shape_.emplace_back(shape, nullptr);
  item.second = std::make_shared<dvm::ShapeRef>(item.first);
  return item.second.get();
}

void LazyFusionKernelAscend::DumpToFile() {
  const std::string dump_dir = "./lazy_fusion_dump";
  auto dir_path = FileUtils::CreateNotExistDirs(dump_dir);
  if (!dir_path.has_value()) {
    MS_LOG(INFO) << "Failed to create directory: " << dump_dir;
    return;
  }
  std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + ".txt";
  std::string file_path = dir_path.value() + "/" + file_name;
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream of(file_path, std::ios::app);
  if (!of.is_open()) {
    MS_LOG(INFO) << "Open dump file '" << file_path << "' failed!";
    ChangeFileMode(file_path, S_IRUSR);
    return;
  }
  of << dump_buf_.str() << "\n";
  of.close();
  ChangeFileMode(file_path, S_IRUSR);
  dump_buf_.str("");
}

dvm::NDObject *LazyFusionKernelAscend::Input(const BaseTensorPtr &x, bool enable_cast,
                                             const std::optional<ShapeVector> &shape) {
  auto input_type = TransType(x->data_type());
  bool cast_to_fp32 = (enable_cast && input_type == dvm::DType::kBFloat16);
  auto device_addr = x->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto xp = device_addr.get();
  // ops_map_ uses device_address as key, because BaseTensorPtr is not continuous, e.g. A is use by B, BaseTensorPtr
  // of A may be different from BaseTensorPtr of B's input, which will affect the relationship of dvm NDObject.
  auto iter = ops_map_.find(xp);
  if (iter == ops_map_.end()) {
    if (input_used_ == inputs_.size()) {
      inputs_.push_back(new Load());
    }
    auto load = inputs_[input_used_++];
    if (shape == std::nullopt) {
      load->shape = x->shape();  // directly point to BaseTensor shape
    } else {
      auto &item = cached_shape_.emplace_back(shape.value(), nullptr);
      load->shape = item.first;
    }
    auto load_op = dvm::Kernel::Load(nullptr, &(load->shape), input_type);
    auto op = cast_to_fp32 ? Cast(load_op, dvm::DType::kFloat32) : load_op;
    load->op = load_op;
    load->tensor = x;
    ops_map_[xp] = op;
    return op;
  }
  auto op = iter->second;
  op = cast_to_fp32 ? Cast(op, dvm::DType::kFloat32) : op;
  return op;
}

void LazyFusionKernelAscend::Output(const BaseTensorPtr &tensor, dvm::NDObject *obj) {
  auto &store = outputs_.emplace_back(obj, tensor);
  ops_map_[store.dev_addr.get()] = obj;
}

bool LazyFusionKernelAscend::HasTensor(const BaseTensorPtr &x) const {
  auto device_addr = x->device_address();
  if (device_addr == nullptr) {
    return false;
  }
  return ops_map_.find(device_addr.get()) != ops_map_.end();
}

void LazyFusionKernelAscend::Launch() {
  MS_LOG(INFO) << "Run launch task dvm kernel start, kernel id is " << id();
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask,
                                     "FlushEager", false);
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  auto stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  if (profiler::Profiler::GetInstance(kAscendDevice)->GetEnableFlag()) {
    EagerMsProfLaunch(stream_ptr);
  } else {
    EagerLaunch(stream_ptr);
  }
  if (LazyFusionFlags::GetInstance().synchronize && !device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for dvm kernel";
  }
  if (!cross_stream_addrs_.empty()) {
    auto &ms = device::MultiStreamController::GetInstance();
    ms->Refresh(device_context_);
    auto task_id_on_stream = ms->LaunchTaskIdOnStream(device_context_, stream_id_);
    ms->RecordEvent(device_context_, task_id_on_stream, stream_id_, cross_stream_addrs_);
    cross_stream_addrs_.clear();
  }
  Clear();
  MS_LOG(INFO) << "Run launch task dvm kernel end, kernel id is " << id();
}

void LazyFusionKernelAscend::Flush() {
  if (outputs_.empty()) {
    Clear();
    return;
  }
  // Async
  auto task = std::make_shared<runtime::PyBoostDeviceTask>([this]() {
    MS_LOG(INFO) << "Run device task dvm kernel start, kernel id is " << id();
    {
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         "MallocIO", false);
      reloc_entry_.reserve(input_used_);
      // Malloc for input tensors
      for (size_t i = 0; i < input_used_; ++i) {
        auto input = inputs_[i];
        runtime::DeviceAddressUtils::MallocForInput(device_context_, input->tensor, false);
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(input->tensor->device_address());
        MS_EXCEPTION_IF_NULL(device_address);
        auto storage_info = device_address->GetTensorStorageInfo();
        auto offset_addr = storage_info ? storage_info->storage_offset * input->tensor->data().itemsize() : 0;
        auto dev_mem = device_address->GetMutablePtr();
        reloc_entry_.emplace_back(input->op, static_cast<void *>(static_cast<uint8_t *>(dev_mem) + offset_addr));
        auto stream_id = device_address->stream_id();
        if (stream_id_ != stream_id) {  // to do: public and use runtime::DeviceAddressUtils::GetCrossStreamAddressInfo
          cross_stream_addrs_.emplace_back(stream_id, dev_mem);
        }
      }
      // Malloc for output tensors
      bool has_store = false;
      for (auto &out : outputs_) {
        auto &device_address = out.dev_addr;
        if (device_address.use_count() == 1 && device_address->address_common()->pointer_ref_count_.use_count() == 1) {
          continue;
        }
        if (device_address->GetPtr() == nullptr) {
          device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative",
                                                         device::tracker::MemType::kPyNativeOutput,
                                                         device_address->GetSize(), device_address.get());
          if (!device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
            MS_LOG(EXCEPTION) << "Allocate memory failed for dvm kernel output";
          }
        }
        auto storage_info = device_address->GetTensorStorageInfo();
        auto offset = storage_info == nullptr
                        ? 0
                        : storage_info->storage_offset * GetTypeByte(TypeIdToType(device_address->type_id()));
        auto dev_mem = device_address->GetMutablePtr();
        auto dst_type = TransType(device_address->type_id());
        dvm::Kernel::Store(static_cast<void *>(static_cast<uint8_t *>(dev_mem) + offset),
                           dst_type != GetDType(out.op) ? Cast(out.op, dst_type) : out.op);
        has_store = true;
      }
      if (!has_store) {
        MS_LOG(INFO) << "Skip launch task dvm kernel, kernel id is " << id();
        Clear();
        return;
      }
    }
    static auto simu = !common::GetEnv(kSimulationLevel).empty();
    if (!simu) {
      // Codegen
      {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostDeviceTask, "CodeGen", false);
        if (LazyFusionFlags::GetInstance().dump_as_text) {
          dump_buf_ << "[lazy_fusion before split] "
                    << "kernel id : " << id() << "\n";
          dump_buf_ << Dump() << "\n";
          EagerCodeGen(reloc_entry_.data(), reloc_entry_.size());
          dump_buf_ << "[lazy_fusion after split] "
                    << "kernel id : " << id() << "\n";
          dump_buf_ << Dump() << "\n";
          dump_buf_ << Das() << "\n";
          DumpToFile();
        } else {
          EagerCodeGen(reloc_entry_.data(), reloc_entry_.size());
        }
      }
      // Launch
      runtime::OpExecutor::DispatchLaunchTask([this]() { Launch(); });
    }
    MS_LOG(INFO) << "Run device task dvm kernel end, kernel id is " << id();
  });
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
  runtime::Pipeline::Get().backend_stage()->runtime::AsyncRQueue::Push(task);  // No flush here
}
}  // namespace kernel
}  // namespace mindspore
