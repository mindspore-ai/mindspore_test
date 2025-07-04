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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TILE_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TILE_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace tile {

class TileAscend : public AclnnKernelMod {
 public:
  TileAscend() : AclnnKernelMod(std::move("aclnnRepeat")) {}
  ~TileAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const { return {1}; }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  void GetAdaptedMultiples(KernelTensor *x_tensor, KernelTensor *multiples_tensor);

  std::vector<int64_t> multiples_;
};
}  // namespace tile
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TILE_ACLNN_KERNEL_MOD_H_
