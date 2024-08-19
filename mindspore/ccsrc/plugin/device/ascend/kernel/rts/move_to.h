/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_MOVE_TO_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_MOVE_TO_H

#include "plugin/device/ascend/kernel/rts/rt_kernel.h"
#include "runtime/device/gsm/swap_manager.h"

namespace mindspore {
namespace kernel {
class MoveTo;
using MoveFunc = bool (MoveTo::*)(const KernelTensor *, const KernelTensor *, void *);
class MoveTo : public RtKernel {
 public:
  MoveTo() = default;
  ~MoveTo() override {}

  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 protected:
  // Init
  bool GetToFromValue(const ValuePtr &value);
  bool GetToValue(const AnfNodePtr &anf_node, size_t to_input_index);
  bool UpdateSizeList(const AnfNodePtr &anf_node);

  static int64_t GetTensorDevice(const KernelTensor *tensor);
  static device::SwapManagerPtr GetSwapManager(const KernelTensor *tensor);
  static bool SyncStream(void *stream_ptr);

  // Memory copy
  static bool D2H(void *host_ptr, const void *device_ptr, void *stream_ptr, size_t size);
  static bool H2D(void *device_ptr, const void *host_ptr, void *stream_ptr, size_t size);

  static bool F2H(void *host_ptr, const string &file_name, size_t size, const device::SwapManagerPtr &swap_manager,
                  device::AsyncIOToken *token);
  static bool H2F(const string &file_name, const void *host_ptr, size_t size,
                  const device::SwapManagerPtr &swap_manager, device::AsyncIOToken *token);

  static bool WaitAioFinish(const KernelTensor *tensor);

  // Move Func
  bool MoveFromDToH(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);
  bool MoveFromDToF(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);

  bool MoveFromHToD(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);
  bool MoveFromHToF(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);

  bool MoveFromFToH(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);
  bool MoveFromFToD(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);

  bool EmptyMove(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr);

 private:
  static std::map<std::pair<int64_t, int64_t>, MoveFunc> func_map_;
  int64_t to_;
};
MS_REG_RTKERNEL(moveto, MoveTo);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_MOVE_TO_H