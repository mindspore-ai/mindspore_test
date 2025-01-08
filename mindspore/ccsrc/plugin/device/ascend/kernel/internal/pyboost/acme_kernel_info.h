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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_INFO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_INFO_H_

#include <memory>
#include <vector>
#include <string>

#include "kernel/kernel.h"
#include "ir/value.h"
#include "include/internal.h"
#include "include/common/factory/ms_factory.h"

#include "plugin/device/ascend/kernel/internal/internal_tiling_cache.h"
#include "plugin/device/ascend/kernel/internal/internal_spinlock.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/kernel/internal/pyboost/acme_pyboost_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "kernel/common/pyboost/op_runner.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
using BaseTensorPtr = tensor::BaseTensorPtr;
// 线程安全
class AcmeKernelInfo {
 public:
  AcmeKernelInfo(std::string &&op_name) : kernel_name_(std::move(op_name)) {}

  virtual ~AcmeKernelInfo() = default;

  bool Init(const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs);

  TilingCacheItemPtr GetOrGenerateTiling(const std::vector<BaseTensorPtr> &inputs,
                                         const std::vector<BaseTensorPtr> &outputs);

  void Launch(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<BaseTensorPtr> &inputs,
              const TilingCacheItemPtr tilingptr);

  void CallAcmeOp(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<BaseTensorPtr> &inputs, uint64_t key);

  virtual void Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) = 0;

 protected:
  virtual internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) {
    return nullptr;
  }

  std::string kernel_name_;
  internal::InternalOpPtr acme_op_{nullptr};
  inline static std::unordered_map<uint64_t, internal::InternalOpPtr> hash_map_;
  // resize by create
  internal::ShapeInfoList acme_inputs_shape_;
  internal::ShapeInfoList acme_outputs_shape_;

  internal::InputsAddrList acme_inputs_addr_;
  internal::OutputsAddrList acme_outputs_addr_;
  internal::WsAddrList acme_wss_addr_;
  std::vector<size_t> workspace_size_list_;

 private:
  void UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor);
  void UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> &arginfos,
                              const std::vector<BaseTensorPtr> &tensorlist);
  void TransAcmeShapes(acme::ShapeInfoList &shapelist, const std::vector<BaseTensorPtr> &tensorlist);
  void UpdateAddr(std::vector<acme::RawDeviceAddr> &addrlist,
                  const std::vector<BaseTensorPtr> &tensorlist);
  void MallocWorkspace(const device::DeviceContext *device_context, size_t stream_id);
  void FreeWorkspace(const device::DeviceContext *device_context);
  SimpleSpinLock lock_;
};

#define MS_ACME_KERNEL_INFO_FACTORY_REG(PRIM_NAME_STR, ACME_NAME_VAR, DERIVE) \
  MS_KERNEL_FACTORY_REG(AcmeKernelInfo, PRIM_NAME_STR, DERIVE);               \
  static const NameMappingRegistrar g_##PRIM_NAME_STR##_ms_to_acme_pyboost_mapper(#PRIM_NAME_STR, ACME_NAME_VAR);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_MOD_H_
