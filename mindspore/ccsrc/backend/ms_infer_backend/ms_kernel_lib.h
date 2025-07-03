/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H__

#include <string>

#include "dalang/dart/runtime/kernel_lib.h"

#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

const char kMindsporeKernelLibName[] = "Mindspore";

class MindsporeKernelLib : public da::runtime::KernelLib {
 public:
  MindsporeKernelLib() : KernelLib(kMindsporeKernelLibName) {
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
       MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    device_context_->Initialize();
  }

  bool RunTensor(da::tensor::DATensor *tensorNode, da::runtime::MemoryPool *mempool = nullptr) const override;

 private:
  device::DeviceContext *device_context_{nullptr};
};

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_KERNEL_LIB_H__
