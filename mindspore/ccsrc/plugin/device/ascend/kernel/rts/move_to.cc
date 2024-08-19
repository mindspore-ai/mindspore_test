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

#include "plugin/device/ascend/kernel/rts/move_to.h"

#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "runtime/device/gsm/swap_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/log_adapter.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kToInputIndex = 2;
constexpr int64_t kNpuInt = 0;
constexpr int64_t kCpuInt = 1;
constexpr int64_t kDiskInt = 2;

static const std::map<std::string, int64_t> ToStrMap{{kToNup, kNpuInt}, {kToCpu, kCpuInt}, {kToDisk, kDiskInt}};

std::map<std::pair<int64_t, int64_t>, MoveFunc> MoveTo::func_map_ = {
  {{kNpuInt, kCpuInt}, &MoveTo::MoveFromDToH},  {{kNpuInt, kDiskInt}, &MoveTo::MoveFromDToF},
  {{kNpuInt, kNpuInt}, &MoveTo::EmptyMove},     {{kCpuInt, kNpuInt}, &MoveTo::MoveFromHToD},
  {{kCpuInt, kDiskInt}, &MoveTo::MoveFromHToF}, {{kCpuInt, kCpuInt}, &MoveTo::EmptyMove},
  {{kDiskInt, kCpuInt}, &MoveTo::MoveFromFToH}, {{kDiskInt, kNpuInt}, &MoveTo::MoveFromFToD},
  {{kDiskInt, kDiskInt}, &MoveTo::EmptyMove}};

bool MoveTo::GetToFromValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<StringImm>()) {
    MS_LOG(ERROR) << "The value of the second input of MoveTo[" << value->ToString() << "] is not a string.";
    return false;
  }
  const auto &str_value = value->cast<StringImmPtr>()->value();
  const auto &iter = ToStrMap.find(str_value);
  if (iter == ToStrMap.end()) {
    MS_LOG(ERROR) << "Invalid value for second input of MoveTo: " << str_value;
    return false;
  }
  to_ = iter->second;
  return true;
}

bool MoveTo::GetToValue(const AnfNodePtr &anf_node, size_t to_input_index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(to_input_index), 0, true);
  auto to_input = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(to_input);
  if (!to_input->isa<ValueNode>()) {
    MS_LOG(ERROR) << "Get to value failed, the second input of MoveTo is not a ValueNode.";
    return false;
  }
  auto to_value_node = to_input->cast<ValueNodePtr>();
  auto to_value = to_value_node->value();
  if (!GetToFromValue(to_value)) {
    MS_LOG(ERROR) << anf_node->fullname_with_scope()
                  << ": GetToValue failed, second input value: " << to_value->ToString();
    return false;
  }
  return true;
}

bool MoveTo::UpdateSizeList(const AnfNodePtr &anf_node) {
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  primitive_ = prim;
  kernel_name_ = prim->name();
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(ERROR) << "#dmsg#Kernel build failed:#dmsg#rts kernel op[" << cnode->fullname_with_scope()
                    << "] Resize failed.";
      return false;
    }
  }
  return true;
}

bool MoveTo::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!GetToValue(anf_node, kToInputIndex)) {
    return false;
  }
  return UpdateSizeList(anf_node);
}

int64_t MoveTo::GetTensorDevice(const KernelTensor *tensor) {
  if (tensor->device_ptr() != nullptr) {
    return kNpuInt;
  }
  const auto &hete_info = tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(hete_info);

  if (hete_info->host_ptr_ != nullptr) {
    return kCpuInt;
  }
  if (!hete_info->file_name_.empty()) {
    return kDiskInt;
  }
  MS_LOG(EXCEPTION) << "Get kenrel tensor device failed.";
}

device::SwapManagerPtr MoveTo::GetSwapManager(const KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &device_name = tensor->device_name();
  const auto device_id = tensor->device_id();
  const auto device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  return device_context->device_res_manager_->swap_manager();
}

bool MoveTo::SyncStream(void *stream_ptr) { return CALL_ASCEND_API(aclrtSynchronizeStream, stream_ptr) == ACL_SUCCESS; }

bool MoveTo::WaitAioFinish(const KernelTensor *tensor) {
  const auto &hete_info = tensor->heterogeneous_info();
  if (hete_info == nullptr) {
    return true;
  }
  if (!hete_info->aio_token_.has_value()) {
    return true;
  }
  const auto token = hete_info->aio_token_.value();
  const auto &swap_manager = GetSwapManager(tensor);
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (!swap_manager->WaitAsyncIO(token)) {
    MS_LOG(ERROR) << "Wait async io failed, token: " << token;
    return false;
  }
  hete_info->aio_token_ = std::nullopt;
  return true;
}

bool MoveTo::D2H(void *host_ptr, const void *device_ptr, void *stream_ptr, size_t size) {
  const auto status =
    CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Moveto kernel aclrtMemcpyAsync device to host failed! src ptr: " << device_ptr
                  << ", dst ptr: " << host_ptr << ", size: " << size << ", stream: " << stream_ptr;
    return false;
  }
  return true;
}

bool MoveTo::H2D(void *device_ptr, const void *host_ptr, void *stream_ptr, size_t size) {
  const auto status =
    CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Moveto kernel aclrtMemcpyAsync host to device failed! src ptr: " << device_ptr
                  << ", dst ptr: " << host_ptr << ", size: " << size << ", stream: " << stream_ptr;
    return false;
  }
  return true;
}

bool MoveTo::F2H(void *host_ptr, const string &file_name, size_t size, const device::SwapManagerPtr &swap_manager,
                 device::AsyncIOToken *token) {
  MS_EXCEPTION_IF_NULL(swap_manager);
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Empty source file name.";
    return false;
  }
  if (!swap_manager->FileToHostMemory(host_ptr, file_name, size, true, token)) {
    MS_LOG(ERROR) << "Moveto kernel FileToHostMemory failed! src ptr: " << host_ptr << ", dst file: " << file_name
                  << ", size: " << size;
    return false;
  }
  return true;
}

bool MoveTo::H2F(const string &file_name, const void *host_ptr, size_t size, const device::SwapManagerPtr &swap_manager,
                 device::AsyncIOToken *token) {
  MS_EXCEPTION_IF_NULL(swap_manager);
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (file_name.empty()) {
    MS_LOG(ERROR) << "Empty dst file name.";
    return false;
  }
  if (!swap_manager->HostMemoryToFile(file_name, host_ptr, size, true, token)) {
    MS_LOG(ERROR) << "Moveto kernel HostMemoryToFile failed! src file name: " << file_name
                  << ", dst host ptr: " << host_ptr << ", size: " << size;
    return false;
  }
  return true;
}

bool MoveTo::MoveFromDToH(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src device ptr.
  const auto device_ptr = src_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);

  // Get dst host ptr.
  const auto &hete_info = dst_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(hete_info);
  const auto &host_ptr = hete_info->host_ptr_;
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Memory copy.
  const auto size = src_tensor->size();
  return D2H(host_ptr, device_ptr, stream_ptr, size);
}

bool MoveTo::MoveFromHToD(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src host ptr.
  const auto &hete_info = src_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(hete_info);
  const auto &host_ptr = hete_info->host_ptr_;
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Get dst device ptr.
  const auto device_ptr = dst_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);

  // Memory copy.
  const auto size = src_tensor->size();
  return H2D(device_ptr, host_ptr, stream_ptr, size);
}

bool MoveTo::MoveFromFToH(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src file name.
  const auto &src_hete_info = src_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(src_hete_info);
  const auto &file_name = src_hete_info->file_name_;

  // Get dst host ptr.
  const auto &dst_hete_info = dst_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(dst_hete_info);
  const auto host_ptr = dst_hete_info->host_ptr_;
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Memory copy.
  const auto size = src_tensor->size();
  const auto &swap_manager = GetSwapManager(dst_tensor);
  MS_EXCEPTION_IF_NULL(swap_manager);
  device::AsyncIOToken token;
  return F2H(host_ptr, file_name, size, swap_manager, &token);
}

bool MoveTo::MoveFromHToF(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src host ptr.
  const auto &src_hete_info = src_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(src_hete_info);
  const auto &host_ptr = src_hete_info->host_ptr_;
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Get dst file name.
  const auto &dst_hete_info = dst_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(dst_hete_info);
  const auto &file_name = dst_hete_info->file_name_;

  // Memory copy.
  const auto size = src_tensor->size();
  const auto &swap_manager = GetSwapManager(dst_tensor);
  MS_EXCEPTION_IF_NULL(swap_manager);
  device::AsyncIOToken token;
  if (!H2F(file_name, host_ptr, size, swap_manager, &token)) {
    return false;
  }
  dst_hete_info->aio_token_ = token;
  return true;
}

bool MoveTo::MoveFromFToD(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src file name.
  const auto &src_hete_info = src_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(src_hete_info);
  const auto &file_name = src_hete_info->file_name_;

  // Get dst device ptr.
  const auto device_ptr = dst_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);

  // Allocate host memory.
  const auto &swap_manager = GetSwapManager(dst_tensor);
  MS_EXCEPTION_IF_NULL(swap_manager);
  const auto size = src_tensor->size();
  const auto host_ptr = swap_manager->AllocHostMemory(size);
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Memory copy.
  device::AsyncIOToken token;
  if (!F2H(host_ptr, file_name, size, swap_manager, &token) || !swap_manager->WaitAsyncIO(token)) {
    return false;
  }

  if (!H2D(device_ptr, host_ptr, stream_ptr, size)) {
    return false;
  }

  // Free host memory
  swap_manager->FreeHostMemory(host_ptr);
  return true;
}

bool MoveTo::MoveFromDToF(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src device ptr.
  const auto device_ptr = src_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);

  // Get dst file name.
  const auto &dst_hete_info = dst_tensor->heterogeneous_info();
  MS_EXCEPTION_IF_NULL(dst_hete_info);
  const auto &file_name = dst_hete_info->file_name_;

  // Allocate host memory.
  const auto &swap_manager = GetSwapManager(dst_tensor);
  MS_EXCEPTION_IF_NULL(swap_manager);
  const auto size = src_tensor->size();
  const auto host_ptr = swap_manager->AllocHostMemory(size);
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Memory copy.
  if (!D2H(host_ptr, device_ptr, stream_ptr, size)) {
    return false;
  }
  if (!SyncStream(stream_ptr)) {
    MS_LOG(ERROR) << "Sync stream during move from device to file failed.";
    return false;
  }
  device::AsyncIOToken token;
  if (!H2F(file_name, host_ptr, size, swap_manager, &token)) {
    return false;
  }
  dst_hete_info->aio_token_ = token;

  // Free host memory
  swap_manager->FreeHostMemory(host_ptr);
  return true;
}

bool MoveTo::EmptyMove(const KernelTensor *, const KernelTensor *, void *) {
  MS_LOG(INFO) << "Kernel tensor has already been stored in target device, skip moving it.";
  return true;
}

bool MoveTo::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  const auto input = inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  const auto output = outputs[0];
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  // Wait unfinished io
  if (!WaitAioFinish(input)) {
    MS_LOG(ERROR) << "Wait async io finish failed for input of kernel: " << kernel_name_;
    return false;
  }
  if (!WaitAioFinish(output)) {
    MS_LOG(ERROR) << "Wait async io finish failed for output of kernel: " << kernel_name_;
    return false;
  }

  const int from = GetTensorDevice(input);
  const auto &func_iter = func_map_.find(std::make_pair(from, to_));
  if (func_iter == func_map_.end()) {
    MS_LOG(ERROR) << "Not supported moving, from: " << from << ", to " << to_;
    return false;
  }
  auto func = func_iter->second;
  return (this->*func)(output, input, stream_ptr);
}
}  // namespace kernel
}  // namespace mindspore
