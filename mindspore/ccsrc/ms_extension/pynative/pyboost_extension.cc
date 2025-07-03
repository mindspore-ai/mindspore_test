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

#include "ms_extension/pynative/pyboost_extension.h"

#ifndef _MSC_VER
#include <cxxabi.h>
#endif

#include "ms_extension/common/tensor.h"
#include "mindspore/ccsrc/include/common/utils/tensor_utils.h"
#include "mindspore/ccsrc/runtime/hardware/device_context.h"
#include "mindspore/ccsrc/runtime/pynative/op_runner.h"
#include "mindspore/core/include/utils/ms_utils.h"
#include "mindspore/ccsrc/runtime/device/device_address_utils.h"
#include "mindspore/ccsrc/include/common/runtime_conf/runtime_conf.h"
#include "mindspore/core/include/utils/ms_context.h"

namespace ms {
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

namespace inner {
std::string GetFunctionName(const char *name) {
#ifdef _MSC_VER
  return name;
#else
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

void SetPromise(const std::string &, const std::tuple<mindspore::stub::StubNodePtr> &tuple, const ms::Tensor &output) {
  mindspore::tensor::SetPromise(tuple, output.tensor());
}

MemBlock::MemBlock(const mindspore::device::DeviceContext *device_context, size_t size, uint32_t stream_id) {
  ptr_ = device_context->device_res_manager_->AllocateMemory(size, stream_id);
  if (ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Alloc workspace failed, size:" << size << ", stream_id:" << stream_id;
  }
  device_context_ = device_context;
}

MemBlock::~MemBlock() { device_context_->device_res_manager_->FreeMemory(ptr_); }

std::string GetDeviceTarget() {
  auto msctx = mindspore::MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(msctx);
  return msctx->get_param<std::string>(mindspore::MsCtxParam::MS_CTX_DEVICE_TARGET);
}
}  // namespace inner

namespace pynative {
void PyboostRunner::Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {
  _inputs_ = inputs;
  _outputs_ = outputs;
  this->_Run();
}

void PyboostRunner::_Run() {
  _device_context_ = mindspore::runtime::OpRunner::GetDeviceContext(inner::GetDeviceTarget());
  this->_PrepareStream();
  this->_PrepareDeviceAddress();
  PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([runner = shared_from_this()]() {
    static auto simu = mindspore::common::IsCompileSimulation();
    if (simu) {
      return;
    }
    // hold workspace until dispatch launch task
    auto workspace_holder = runner->_MallocDeviceAddress();
    runner->_DispatchLaunchTask();
  }));
}

void PyboostRunner::_PrepareStream() {
  _stream_id_ = static_cast<size_t>(PyBoostUtils::cur_stream_id());
  _stream_ = _device_context_->device_res_manager_->GetStream(_stream_id_);
}

void PyboostRunner::_PrepareDeviceAddress() {
  for (size_t i = 0; i < _inputs_.size(); i++) {
    if (!_inputs_[i].is_defined()) {
      continue;
    }
    mindspore::runtime::DeviceAddressUtils::CreateInputTensorAddress(_device_context_, _stream_id_, i,
                                                                     _inputs_[i].tensor());
  }
  std::vector<mindspore::tensor::TensorPtr> outs;
  outs.reserve(_outputs_.size());
  for (auto &out : _outputs_) {
    if (out.tensor() != nullptr && out.tensor()->device_address() == nullptr) {
      (void)outs.emplace_back(out.tensor());
    }
  }
  mindspore::runtime::DeviceAddressUtils::CreateOutputTensorAddress(_device_context_, _stream_id_, outs);
}

inner::MemBlockPtr PyboostRunner::_MallocDeviceAddress() {
  {
    // input tensors
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyBoostMallocInput,
                                                  mindspore::runtime::ProfilerRecorder::kNoName, false);
    for (auto &inp : _inputs_) {
      if (!inp.is_defined()) {
        continue;
      }
      mindspore::runtime::DeviceAddressUtils::MallocForInput(_device_context_, inp.tensor(), false);
    }
  }
  {
    // output tensors
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyBoostMallocOutput,
                                                  mindspore::runtime::ProfilerRecorder::kNoName, false);
    std::vector<mindspore::tensor::TensorPtr> outs;
    outs.reserve(_outputs_.size());
    for (auto &out : _outputs_) {
      if (out.tensor() != nullptr) {
        (void)outs.emplace_back(out.tensor());
      }
    }
    mindspore::runtime::DeviceAddressUtils::MallocForOutputs(_device_context_, outs);
  }
  // calculate and alloc workspace
  inner::MemBlockPtr ws_mng;
  auto workspace_size = this->CalcWorkspace();
  if (workspace_size > 0) {
    ws_mng = std::make_shared<inner::MemBlock>(_device_context_, workspace_size, _stream_id_);
    this->_workspace_ptr_ = ws_mng->ptr_;
  } else {
    this->_workspace_ptr_ = nullptr;
  }
  return ws_mng;
}

void PyboostRunner::_DispatchLaunchTask() {
  mindspore::runtime::OpExecutor::DispatchLaunchTask([runner = shared_from_this()]() {
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyNativeLaunchTask,
                                                  runner->op_name(), false);
    runner->LaunchKernel();
    if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      if (!runner->_device_context_->device_res_manager_->SyncAllStreams()) {
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << runner->op_name();
      }
    }
  });
}
}  // namespace pynative
}  // namespace ms
