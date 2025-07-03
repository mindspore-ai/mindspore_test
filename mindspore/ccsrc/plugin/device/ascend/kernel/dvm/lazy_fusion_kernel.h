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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <utility>
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_address.h"
#include "plugin/res_manager/ascend/dvm/dvm.h"
#include "mindspore/core/include/ir/tensor.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "runtime/pynative/lazy_fusion.h"

namespace mindspore {
namespace kernel {
using ShapeRefPtr = std::shared_ptr<dvm::ShapeRef>;
using TensorPtr = tensor::TensorPtr;
using OpRunnerPtr = std::shared_ptr<pyboost::OpRunner>;

class LazyFusionQueue : public runtime::AsyncRQueue {
 public:
  LazyFusionQueue(const string &name, runtime::kThreadWaitLevel waitLevel) : AsyncRQueue(name, waitLevel) {}

  void Push(const runtime::AsyncTaskPtr &task) override;
  void Wait() override;
  bool Empty() override;
  void WorkerJoin() override;
  runtime::kThreadWaitLevel GetCurrentLevel();
};

class LazyFusionKernelAscend;
class LazyFusionManager {
 public:
  LazyFusionManager() = default;
  ~LazyFusionManager();

  LazyFusionKernelAscend *Get(const device::DeviceContext *context, size_t stream);

  void Flush();
  bool Empty() { return current_ == nullptr; }

  void FreeKernel(LazyFusionKernelAscend *k) {
    std::lock_guard<std::mutex> guard(mutex_);
    pool_.push(k);
  }

 private:
  LazyFusionKernelAscend *NewKernel();

  std::queue<LazyFusionKernelAscend *> pool_;
  LazyFusionKernelAscend *current_{nullptr};
  std::mutex mutex_;
  std::atomic<size_t> id_{0};
};

extern LazyFusionManager g_lazy_fusion_manager;

class LazyFusionKernelAscend : public dvm::Kernel {
 public:
  LazyFusionKernelAscend();
  ~LazyFusionKernelAscend();
  void Flush();

  void Reset(const device::DeviceContext *context, size_t stream_id) {
    device_context_ = context;
    stream_id_ = stream_id;
  }
  const device::DeviceContext *device_context() const { return device_context_; }
  size_t stream_id() const { return stream_id_; }
  void set_id(size_t id) { id_ = id; }
  size_t id() const { return id_; }

  dvm::NDObject *Input(const TensorPtr &x, bool enable_cast = true,
                       const std::optional<ShapeVector> &shape = std::nullopt);
  void Output(const TensorPtr &tensor, dvm::NDObject *obj);

  TensorPtr Output(dvm::NDObject *obj, TypeId dtype, const ShapeVector &shape) {
    auto tensor = std::make_shared<tensor::Tensor>(dtype, shape);
    runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context_, stream_id_, tensor,
                                                           LongToSize(tensor->data().nbytes()));
    Output(tensor, obj);
    return tensor;
  }

  ShapeVector GetShape(dvm::NDObject *obj) {
    auto shape_ref = dvm::Kernel::GetShape(obj);
    return ShapeVector(shape_ref->data, shape_ref->data + shape_ref->size);
  }

  dvm::ShapeRef *GetShapeRef(const ShapeVector &shape);
  void DumpToFile();

  dvm::DType TransType(TypeId type) {
    switch (type) {
      case kNumberTypeBool:
        return dvm::DType::kBool;
      case kNumberTypeInt32:
        return dvm::DType::kInt32;
      case kNumberTypeFloat16:
        return dvm::DType::kFloat16;
      case kNumberTypeFloat32:
        return dvm::DType::kFloat32;
      case kNumberTypeBFloat16:
        return dvm::DType::kBFloat16;
      default:
        return dvm::DType::kTypeEnd;
    }
  }

  void *AllocWorkspace(uint64_t size);

  bool HasTensor(const TensorPtr &x) const;

 private:
  void Launch();

  void ClearGraph() {
    for (size_t i = 0; i < input_used_; i++) {
      inputs_[i]->tensor.reset();
    }
    ops_map_.clear();
    input_used_ = 0;
    outputs_.clear();
    reloc_entry_.clear();
    cached_shape_.clear();
  }

  void ClearKernel() {
    cross_stream_addrs_.clear();
    EagerClear();
    g_lazy_fusion_manager.FreeKernel(this);
  }

  void Clear() {
    ClearGraph();
    ClearKernel();
  }

  struct Load {
    Load() = default;
    dvm::ShapeRef shape;
    dvm::NDObject *op;
    TensorPtr tensor;
  };

  struct Store {
    Store() = default;
    Store(dvm::NDObject *p, const TensorPtr &t) : op(p) {
      dev_addr = std::static_pointer_cast<device::DeviceAddress>(t->device_address());
      MS_EXCEPTION_IF_NULL(dev_addr);
    }
    dvm::NDObject *op;
    device::DeviceAddressPtr dev_addr;
  };

  std::unordered_map<void *, dvm::NDObject *> ops_map_;
  std::vector<Load *> inputs_;
  std::vector<Store> outputs_;
  std::vector<dvm::RelocEntry> reloc_entry_;
  std::vector<std::pair<uint32_t, void *>> cross_stream_addrs_;
  std::vector<std::pair<ShapeVector, ShapeRefPtr>> cached_shape_;
  size_t input_used_{0};
  std::stringstream dump_buf_;
  const device::DeviceContext *device_context_;
  size_t stream_id_;
  size_t id_{0};
};

static inline void FlushLazyFusion() { g_lazy_fusion_manager.Flush(); }
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H
