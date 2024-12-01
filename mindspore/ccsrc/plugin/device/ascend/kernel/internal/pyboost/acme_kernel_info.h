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
#include "acme/include/acme.h"
#include "acme/tiling_mem_mgr.h"
#include "include/common/factory/ms_factory.h"

#include "plugin/device/ascend/kernel/internal/acme/acme_tiling_cache.h"
#include "plugin/device/ascend/kernel/internal/acme/acme_spinlock.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace kernel {
// 线程安全
class AcmeKernelInfo {
 public:
  AcmeKernelInfo(const std::string &name) { kernel_name_ = name; }

  virtual ~AcmeKernelInfo() = default;

  virtual bool Init(const std::vector<tensor::BaseTensorPtr> &inputs,
                    const std::vector<tensor::BaseTensorPtr> &outputs);

  virtual TilingCacheItemPtr GetOrGenerateTiling(const std::vector<tensor::BaseTensorPtr> &inputs,
                                                 const std::vector<tensor::BaseTensorPtr> &outputs);

  virtual bool Launch(const device::DeviceContext *device_context, const TilingCacheItemPtr tilingptr,
                      const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                      void *stream_ptr);

 protected:
  virtual acme::AcmeOpPtr CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                       const acme::OutputsImmutableInfoList &outputs,
                                       const std::vector<tensor::BaseTensorPtr> &ms_inputs,
                                       const std::vector<tensor::BaseTensorPtr> &ms_outputs) {
    return nullptr;
  }

  acme::AcmeOpPtr acme_op_{nullptr};
  // resize by create
  acme::ShapeInfoList acme_inputs_shape_;
  acme::ShapeInfoList acme_outputs_shape_;

  acme::InputsAddrList acme_inputs_addr_;
  acme::OutputsAddrList acme_outputs_addr_;
  acme::WsAddrList acme_wss_addr_;
  std::vector<size_t> workspace_size_list_;

 private:
  void UpdateArgImmutableInfo(ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor);
  void UpdateArgImmutableInfo(const std::vector<tensor::BaseTensorPtr> &tensorlist,
                              std::vector<acme::ArgImmutableInfo> &arginfos);

  void UpdateAddr(const std::vector<tensor::BaseTensorPtr> &tensorlist, acme::InputsAddrList &addrlist);

  void UpdateAddr(const std::vector<tensor::BaseTensorPtr> &inputs, const std::vector<tensor::BaseTensorPtr> &outputs);

  void MallocWorkspace(const device::DeviceContext *device_context);
  void FreeWorkspace(const device::DeviceContext *device_context);
  std::string kernel_name_;
  SimpleSpinLock lock_;
};

std::shared_ptr<AcmeKernelInfo> GetAcmeKernelInfo(
  const std::string &kernelname, const std::vector<tensor::BaseTensorPtr> &ms_inputs,
  const std::vector<tensor::BaseTensorPtr> &ms_outputs,
  std::unordered_map<uint64_t, std::shared_ptr<AcmeKernelInfo>> &hash_map) {
  std::shared_ptr<AcmeKernelInfo> kernel_info = nullptr;
  static std::unordered_map < uint64_t, std::shared_ptr<AcmeKernelInfo> hash_map_;
  auto op_key = CalcAcmeOpApiHash(kernelname, ms_inputs, ms_outputs);
  auto it = hash_map_.find(op_key);
  if (it != hash_map_.end()) {
    kernel_info = it->second;
  } else {
    if (Factory<AcmeKernelInfo>::Instance().IsRegistered(kernelname)) {
      MS_LOG(INFO) << "Supported by Acme Op: " << kernelname;
      kernel_info = std::static_pointer_cast<AcmeKernelInfo>(Factory<AcmeKernelInfo>::Instance().Create(kernelname));
    }

    if (kernel_info == nullptr) {
      MS_LOG(WARNING) << "Acme can't find op[" << kernelname << "]";
      return nullptr;
    }

    auto ret = kernel_info->Init();
    if (!ret) {
      return nullptr;
    }
    hash_map_[op_key] = kernel_info;
  }

  return kernel_info;
}

#define GET_ACMEKERNELINFO(kernel_info, kernelname, ms_inputs, ms_outputs)          \
  do {                                                                              \
    static std::unordered_map<uint64_t, std::shared_ptr<AcmeKernelInfo>> hash_map_; \
    kernel_info = GetAcmeKernelInfo(kernelname, ms_inputs, ms_outputs, hash_map_);  \
  } while (false)

#define MS_ACME_KERNEL_INFO_FACTORY_REG(PRIM_NAME_STR, ACME_NAME_VAR, DERIVE) \
  MS_KERNEL_FACTORY_REG(AcmeKernelInfo, PRIM_NAME_STR, DERIVE);               \
  static const NameMappingRegistrar g_##PRIM_NAME_STR##_ms_to_acme_pyboost_mapper(#PRIM_NAME_STR, ACME_NAME_VAR);

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_MOD_H_
