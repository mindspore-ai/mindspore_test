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
#ifndef MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDBASE_H_
#define MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDBASE_H_

#include <memory>
#include <string>
#include <map>
#include "mindspore/core/include/base/base.h"
#include "mindspore/core/include/base/base_ref.h"
#include "backend/backend_manager/visible.h"
#include "backend/backend_manager/backend_jit_config.h"
#include "ir/tensor.h"
#include "include/common/pynative/op_runner_info.h"

namespace mindspore {
namespace backend {
using BackendGraphId = uint32_t;

// The return value enum of BackendBase::Run.
enum RunningStatus {
  kRunningSuccess = 0,
  kRunningFailure,
};

enum IRFormat {
  kAir = 0,
};

using IsPyBoostRegisteredFunc = std::function<bool(const std::string &device_target, const std::string &op_name)>;
using RunPyBoostCallFunc = std::function<void(runtime::OpRunnerInfo *, VectorRef *)>;

// The base class of all supported backend.
class BACKEND_MANAGER_EXPORT BackendBase {
 public:
  // The backend graph Build interface, the return value is the built graph id.
  virtual BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) = 0;

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  virtual RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) = 0;

  // convert mindir to ir_format
  virtual void ConvertIR(const FuncGraphPtr &anf_graph,
                         const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                         IRFormat ir_format) {
    return;
  }

  // export graph to ir_format. If is_save_to_file=True, save as file; if False, return as string
  virtual std::string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                               IRFormat ir_format) {
    return "";
  }

  // clear the resource, init is in constructor function
  virtual void Clear() {}

  virtual void SetPyBoostRegistered(const IsPyBoostRegisteredFunc &func, const RunPyBoostCallFunc &call_func) {}
};

using BackendBasePtr = std::shared_ptr<BackendBase>;
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDBASE_H_
