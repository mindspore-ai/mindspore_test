/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_

#include <memory>
#include <string>
#include <map>
#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/res_manager/memory_manager.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/collective/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include "backend/ge_backend/executor/ge_graph_executor.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeGraphExecutor;
class GeKernelExecutor;
class AscendDeviceResManager;
// The Ascend device properties defined by MindSpore because ACL does not have interface to get this info.
struct AscendDeviceProperties {
  std::string name;
  size_t total_memory;
  size_t free_memory;
};

class GeDeviceContext
    : public DeviceInterface<backend::ge_backend::GeGraphExecutor, GeKernelExecutor, AscendDeviceResManager> {
 public:
  explicit GeDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~GeDeviceContext() override = default;

  void Initialize() override;

  void ContextInitGe() const;

  void Destroy() override;

  bool PartitionGraph(const FuncGraphPtr &func_graph) const override;
  RunMode GetRunMode(const FuncGraphPtr &func_graph) const override;

  DeprecatedInterface *GetDeprecatedInterface() override;

  static uint32_t GetDeviceCount();
  static std::string GetDeviceName(uint32_t);
  static AscendDeviceProperties GetDeviceProperties(uint32_t);

  uint32_t GetExecuteTimeout() override;
  std::string GetAoeJobType() override;
  std::string GetPrecisionMode() override;

 private:
  DISABLE_COPY_AND_ASSIGN(GeDeviceContext);

  void InitDump() const;

  std::unique_ptr<AscendDeprecatedInterface> deprecated_interface_;
  mutable bool ge_initialized_{false};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_
