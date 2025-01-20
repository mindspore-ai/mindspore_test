/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INFO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INFO_H_

#include <memory>
#include <vector>
#include <string>

#include "kernel/kernel.h"
#include "ir/value.h"
#include "include/internal.h"
#include "plugin/device/ascend/kernel/internal/tiling_mem_mgr.h"
#include "include/common/factory/ms_factory.h"

#include "plugin/device/ascend/kernel/internal/internal_tiling_cache.h"
#include "plugin/device/ascend/kernel/internal/internal_spinlock.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/kernel/internal/internal_helper.h"
#include "plugin/device/ascend/kernel/internal/pyboost/internal_pyboost_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "kernel/common/pyboost/op_runner.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
using BaseTensorPtr = tensor::BaseTensorPtr;
// 线程安全
class InternalKernelInfo {
 public:
  InternalKernelInfo(std::string &&op_name) : kernel_name_(std::move(op_name)) {}
  virtual ~InternalKernelInfo() = default;

  bool Init(const ValuePtrList &input_values, std::vector<BaseTensorPtr> &inputs, std::vector<BaseTensorPtr> &outputs,
            const std::vector<BaseTensorPtr> &op_outputs);
  void GetOrCreateKernel(const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs,
                         uint64_t key);
  virtual void Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) = 0;

  static void UpdateAddr(std::vector<internal::RawDeviceAddr> &addrlist, const std::vector<BaseTensorPtr> &tensorlist) {
    addrlist.resize(tensorlist.size());
    for (size_t i = 0; i < tensorlist.size(); i++) {
      if (tensorlist[i] == nullptr) {
        addrlist[i] = nullptr;
      } else {
        addrlist[i] = tensorlist[i]->device_address()->GetMutablePtr();
      }
    }
  }

  static void MallocWorkspace(const device::DeviceContext *device_context, size_t stream_id,
                              const internal::InternalOpPtr &internal_op, internal::WsAddrList &internal_wss_addr) {
    auto workspace_size_list = internal_op->GetWorkspaceSize();
    internal_wss_addr.resize(workspace_size_list.size());
    for (size_t i = 0; i < workspace_size_list.size(); i++) {
      auto ptr = device_context->device_res_manager_->AllocateMemory(workspace_size_list[i], stream_id);
      if (ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Alloc failed, size:" << workspace_size_list[i] << ", stream_id:" << stream_id;
      }
      internal_wss_addr[i] = ptr;
    }
  }

  static void FreeWorkspace(const device::DeviceContext *device_context, internal::WsAddrList &internal_wss_addr) {
    for (size_t i = 0; i < internal_wss_addr.size(); i++) {
      device_context->device_res_manager_->FreeMemory(internal_wss_addr[i]);
      internal_wss_addr[i] = nullptr;
    }
  }

 protected:
  TilingCacheItemPtr GetOrGenerateTiling(const std::vector<BaseTensorPtr> &inputs);

  void GetInputAndOutputIndex(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList &input_values);

  virtual internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                               const internal::OutputsImmutableInfoList &outputs) {
    return nullptr;
  }

  std::string kernel_name_;
  internal::InternalOpPtr internal_op_{nullptr};
  inline static std::unordered_map<uint64_t, internal::InternalOpPtr> hash_map_;
  std::vector<size_t> ms_inputs_idx_list_;
  std::vector<size_t> ms_outputs_idx_list_;
  internal::ShapeInfoList internal_inputs_shape_;
  internal::ShapeInfoList internal_outputs_shape_;
  internal::InputsImmutableInfoList inputs_ii_;
  internal::OutputsImmutableInfoList outputs_ii_;
  TilingCacheItemPtr tiling_info_{nullptr};

 private:
  void UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor);
  void UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> &arginfos,
                              const std::vector<BaseTensorPtr> &tensorlist);
  void TransInternalShapes(internal::ShapeInfoList &shapelist, const std::vector<BaseTensorPtr> &tensorlist);
  SimpleSpinLock lock_;
};

#define MS_INTERNAL_KERNEL_INFO_FACTORY_REG(PRIM_NAME_STR, INTERNAL_NAME_VAR, DERIVE) \
  MS_KERNEL_FACTORY_REG(InternalKernelInfo, PRIM_NAME_STR, DERIVE);                   \
  static const NameMappingRegistrar g_##PRIM_NAME_STR##_ms_to_acme_pyboost_mapper(#PRIM_NAME_STR, INTERNAL_NAME_VAR);

#define LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_)                               \
  do {                                                                                                               \
    const std::string kernel_name = kernel_name_;                                                                    \
    internal::InternalOpPtr internal_op = internal_op_;                                                              \
    std::vector<BaseTensorPtr> inputs_ = inputs;                                                                     \
    std::vector<BaseTensorPtr> outputs_ = outputs;                                                                   \
    TilingCacheItemPtr tiling_ptr = tiling_info_;                                                                    \
    pyboost::PyBoostUtils::DispatchRun(                                                                              \
      std::make_shared<runtime::PyBoostDeviceTask>([kernel_name, op, internal_op, inputs_, outputs_, tiling_ptr]() { \
        MS_LOG(INFO) << "Launch InternalKernel " << kernel_name << " start";                                         \
        auto device_context = op->device_context();                                                                  \
        for (auto &input : inputs_) {                                                                                \
          if (input != nullptr) {                                                                                    \
            pyboost::PyBoostUtils::MallocOpInputs(device_context, input);                                            \
          }                                                                                                          \
        }                                                                                                            \
        pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs_);                                            \
        internal::InputsAddrList inputs_addr;                                                                        \
        internal::OutputsAddrList outputs_addr;                                                                      \
        InternalKernelInfo::UpdateAddr(inputs_addr, inputs_);                                                        \
        InternalKernelInfo::UpdateAddr(outputs_addr, outputs_);                                                      \
        internal::WsAddrList internal_wss_addr;                                                                      \
        runtime::Pipeline::Get().launch_stage()->Wait();                                                             \
        InternalKernelInfo::MallocWorkspace(device_context, op->stream_id(), internal_op, internal_wss_addr);        \
        internal_op->SetTilingInfo(tiling_ptr->tiling_info_);                                                        \
        auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());                           \
        internal::InternalStatus status =                                                                            \
          internal_op->Launch(inputs_addr, outputs_addr, internal_wss_addr, stream_ptr, kernel_name);                \
        InternalKernelInfo::FreeWorkspace(device_context, internal_wss_addr);                                        \
        InternalTilingCache::GetInstance().Unbind(tiling_ptr);                                                       \
        if (status != internal::InternalStatus::kInternalOk) {                                                       \
          MS_LOG(EXCEPTION) << "Launch InternalKernel failed, kernel_name: " << kernel_name;                         \
        }                                                                                                            \
        MS_LOG(INFO) << "Launch InternalKernel " << kernel_name << " end";                                           \
      }));                                                                                                           \
  } while (false)

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INFO_H_
