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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
#define MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_

#include <map>
#include <unordered_map>
#include <memory>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "ir/tensor.h"
#include "include/common/visible.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;

class BackwardNode;

struct BackwardNodePreHook {
  virtual ~BackwardNodePreHook() = default;
  virtual void operator()(ValuePtrList *grad) = 0;
};

struct PyTensorBackwardNodePreHook : public BackwardNodePreHook {
  PyTensorBackwardNodePreHook(const py::function &hook_fn, size_t output_idx);
  ~PyTensorBackwardNodePreHook() override;
  void operator()(ValuePtrList *grad) override;
  py::function hook_fn_;
  size_t output_idx_;
};

using CppHookFn = std::function<tensor::TensorPtr(const tensor::TensorPtr &)>;
struct CppTensorBackwardNodePreHook : public BackwardNodePreHook {
  CppTensorBackwardNodePreHook(CppHookFn hook_fn, size_t output_idx);
  void operator()(ValuePtrList *grad) override;
  CppHookFn hook_fn_;
  size_t output_idx_;
};

struct RegisterHook {
  /// \brief Register a backward hook
  ///
  /// \ void
  PYNATIVE_EXPORT static uint64_t RegisterTensorBackwardHook(const tensor::TensorPtr &tensor, const py::function &hook);

  /// \brief Remove a backward hook
  ///
  /// \ void
  PYNATIVE_EXPORT static void RemoveTensorBackwardHookOfGraph(uint64_t tensor_id, uint64_t handle_id);
  PYNATIVE_EXPORT static void RemoveTensorBackwardHook(uint64_t handle_id);
  PYNATIVE_EXPORT static py::list GetHooks(const tensor::TensorPtr &tensor);

  PYNATIVE_EXPORT static unsigned RegisterCppTensorBackwardHook(const tensor::TensorPtr &tensor, const CppHookFn &hook);
  PYNATIVE_EXPORT static void RemoveCppTensorBackwardHook(const tensor::TensorPtr &tensor, unsigned hook_id);

  static void ClearHookMap() { hook_id_node_map_.clear(); }

  // For store hook
  inline static uint64_t unique_id_ = 0;
  inline static std::unordered_map<uint64_t, std::weak_ptr<BackwardNode>> hook_id_node_map_;
  inline static std::unordered_map<uint64_t, std::weak_ptr<std::map<uint64_t, py::function>>> tensor_id_with_hook_map_;
  inline static std::unordered_map<uint64_t, uint64_t> unique_id_with_tensor_id_;
};
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
