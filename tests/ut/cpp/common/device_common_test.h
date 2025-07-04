/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
#define TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H

#include <memory>

#include "common/common_test.h"
#define private public
#define protected public
#include "abstract/abstract_function.h"
#include "runtime/graph_scheduler/control_node_parser.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "common/device_address.h"
#include "common/kernel_tensor.h"
#include "common/kernel_utils.h"
#include "common/common_utils.h"
#include "kernel/framework_utils.h"
#define private public
#define protected public

namespace mindspore {
namespace runtime {
namespace test {
using abstract::AbstractFuncUnion;
using abstract::AbstractTensor;
using abstract::AbstractTensorPtr;
using abstract::AnalysisContext;
using abstract::FuncGraphAbstractClosure;
using device::DeviceAddress;
using device::DeviceAddressPtr;
using device::DeviceContextKey;
using device::DeviceContextRegister;
using device::DeviceType;
using kernel::AddressPtr;
using kernel::KernelTensorPtr;
using session::KernelGraph;

class TestDeviceAddress : public DeviceAddress {
 public:
  TestDeviceAddress() : DeviceAddress() {}
  TestDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  TestDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                    uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  ~TestDeviceAddress() {}
  virtual bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr,
                                bool sync_on_demand) const {
    return true;
  }
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                                const std::string &format) const {
    return true;
  }
  virtual void *GetMutablePtr() const { return nullptr; }
  virtual void ClearDeviceMemory() {}
};

class TestKernelMod : public kernel::KernelMod {
 public:
  TestKernelMod() = default;
  ~TestKernelMod() override = default;
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    return true;
  }
  std::vector<kernel::KernelAttr> GetOpSupport() override { return {}; }
};

class TestDeviceResManager : public device::DeviceResManager {
 public:
  TestDeviceResManager() = default;
  ~TestDeviceResManager() override = default;

  virtual bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const {
    static size_t total_size_{1024};
    MS_EXCEPTION_IF_NULL(address);
    if (address->GetSize() > total_size_) {
      return false;
    }
    total_size_ -= address->GetSize();
    return true;
  }
  virtual void FreeMemory(DeviceAddress *const &address) const {}
  virtual void *AllocateMemory(size_t size, const uint32_t stream_id = UINT32_MAX) const { return nullptr; }
  virtual void FreeMemory(void *const ptr) const {}
  virtual void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                               const std::vector<size_t> &keep_addr_sizes) const {}
  virtual DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format,
                                               TypeId type_id, const ShapeVector &shape,
                                               const UserDataPtr &user_data = nullptr) const {
    return std::make_shared<TestDeviceAddress>(device_ptr, device_size, format, type_id, "CPU", 0);
  }

  virtual DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                               const Format &format, TypeId type_id, const std::string &device_name,
                                               uint32_t device_id, uint32_t stream_id,
                                               const UserDataPtr &user_data = nullptr) const {
    return std::make_shared<TestDeviceAddress>(ptr, size, "falut", type_id, device_name, 0);
  }

  DeviceAddressPtr CreateDeviceAddress() const {
    auto device_address = std::make_shared<TestDeviceAddress>();
    device_address->set_device_name(device_context_->device_context_key().device_name_);
    device_address->set_device_id(device_context_->device_context_key().device_id_);
    return device_address;
  }
};

class TestKernelExecutor : public device::KernelExecutor {
 public:
  TestKernelExecutor() = default;
  ~TestKernelExecutor() override = default;

  virtual void OptimizeGraph(const FuncGraphPtr &graph) const {
    MS_EXCEPTION_IF_NULL(graph);
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto &nodes = kernel_graph->execution_order();
    for (const auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      SetKernelInfo(node);
    }
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    auto pm = std::make_shared<opt::PassManager>();
    pm->AddPass(std::make_shared<opt::AllReduceFusion>());
    optimizer->AddPassManager(pm);
    (void)optimizer->Optimize(kernel_graph);
    kernel_graph->SetExecOrderByDefault();
  }

  virtual void CreateKernel(const std::vector<CNodePtr> &nodes) const {
    for (const auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      SetKernelInfo(node);

      std::vector<size_t> input_size_list;
      std::vector<size_t> output_size_list;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      for (size_t input_index = 0; input_index < input_num; ++input_index) {
        auto [input_node, index] = common::AnfAlgo::GetPrevNodeOutput(node, input_index, true);
        size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, index);
        (void)input_size_list.emplace_back(tensor_size);
        if (AnfAlgo::OutputAddrExist(input_node, index)) {
          continue;
        }
        AnfAlgo::SetOutputAddr(std::make_shared<TestDeviceAddress>(nullptr, tensor_size), index, input_node);
      }
      size_t output_num = AnfAlgo::GetOutputTensorNum(node);
      for (size_t output_index = 0; output_index < output_num; ++output_index) {
        size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(node, output_index);
        (void)output_size_list.emplace_back(tensor_size);
        AnfAlgo::SetOutputAddr(std::make_shared<TestDeviceAddress>(nullptr, tensor_size), output_index, node);
      }

      const size_t kDefaultWorkSpaceSize = 4;
      auto kernel_mod_ptr = std::make_shared<TestKernelMod>();
      kernel_mod_ptr->SetInputSizeList(input_size_list);
      kernel_mod_ptr->SetOutputSizeList(output_size_list);
      kernel_mod_ptr->SetWorkspaceSizeList({kDefaultWorkSpaceSize});
      AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
      AnfAlgo::SetWorkspaceAddr(std::make_shared<TestDeviceAddress>(nullptr, kDefaultWorkSpaceSize), 0, node);
    }
  }

 private:
  void SetKernelInfo(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    if (node->kernel_info() == nullptr) {
      auto kernel_info = std::make_shared<device::KernelInfo>();
      node->set_kernel_info(kernel_info);
    }

    const auto &kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    if (kernel_info->select_kernel_build_info() != nullptr) {
      return;
    }

    std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
    std::vector<std::string> inputs_format;
    std::vector<TypeId> inputs_type;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      (void)inputs_format.emplace_back(kOpFormat_DEFAULT);
      (void)inputs_type.emplace_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    }

    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    size_t output_num = AnfAlgo::GetOutputElementNum(node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      (void)outputs_format.emplace_back(kOpFormat_DEFAULT);
      (void)outputs_type.emplace_back(common::AnfAlgo::GetOutputInferDataType(node, output_index));
    }

    builder->SetOriginDataFormat(kOpFormat_DEFAULT);
    builder->SetInputsFormat(inputs_format);
    builder->SetInputsDeviceType(inputs_type);
    builder->SetOutputsFormat(outputs_format);
    builder->SetOutputsDeviceType(outputs_type);
    kernel_info->set_select_kernel_build_info(builder->Build());
  }
};

class TestGraphExecutor {
 public:
  ~TestGraphExecutor() = default;
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::TensorPtr> &inputs,
                std::vector<tensor::TensorPtr> *outputs, const std::map<string, string> &compile_options) {
    MS_LOG(INFO) << "Ut run test graph.";
    MS_EXCEPTION_IF_NULL(graph);
    const auto &kernel_graph = dynamic_cast<KernelGraph *>(graph.get());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    for (const auto &kernel : kernel_graph->execution_order()) {
      MS_EXCEPTION_IF_NULL(kernel);
      MS_EXCEPTION_IF_NULL(kernel->kernel_info());
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(kernel, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      auto context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_tensor->device_name(), device_tensor->device_id()});
      MS_EXCEPTION_IF_NULL(context);
      context->device_res_manager_->AllocateMemory(device_tensor.get(), device_tensor->stream_id());
      MS_LOG(INFO) << "Alloc memory in run graph";
    }
    return true;
  }
};

class TestDeviceContext : public device::DeviceInterface<TestKernelExecutor, TestDeviceResManager> {
 public:
  explicit TestDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {
    graph_executor_ = std::make_shared<TestGraphExecutor>();
  }
  ~TestDeviceContext() override = default;

  virtual void Initialize() {}
  virtual DeviceType GetDeviceType() const { return DeviceType::kCPU; }
private:
  std::shared_ptr<TestGraphExecutor> graph_executor_;
};

class TestResManager : public device::HalResBase {
 public:
  TestResManager(const device::ResKey &res_key) : device::HalResBase(res_key) {}

  ~TestResManager() override = default;

  DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                                       TypeId type_id, const std::string &device_name, uint32_t device_id,
                                       uint32_t stream_id, const UserDataPtr &user_data = nullptr) const override {
    return std::make_shared<TestDeviceAddress>(ptr, size, "NCHW", type_id, "CPU", 0);
  }
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override {}
  void FreeMemory(void *ptr) const override {}
  void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                       const std::vector<size_t> &keep_addr_sizes) const override {}
};
}  // namespace test
}  // namespace runtime
}  // namespace mindspore
#endif  // TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
