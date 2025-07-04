/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "kernel/gpu/data/dataset_profiling.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "include/backend/data_queue/blocking_queue.h"
namespace mindspore {
namespace kernel {
using mindspore::device::DataQueueItem;

class DatasetIteratorKernelMod : public NativeGpuKernelMod {
 public:
  DatasetIteratorKernelMod();
  ~DatasetIteratorKernelMod();

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 private:
  bool ReadDevice(std::vector<DataQueueItem> *data);
  std::string queue_name_;
  bool is_opened_;
  bool profiling_enable_;
  std::shared_ptr<GetNextProfiling> profiling_op_;
  std::vector<TypeId> types_;
  std::vector<DataQueueItem> output_data_;
  bool dynamic_shape_{false};
};

MS_REG_GPU_KERNEL(GetNext, DatasetIteratorKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_
