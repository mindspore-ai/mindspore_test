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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_ERROR_MANAGER_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_ERROR_MANAGER_H

#include <memory>
#include <vector>
#include "acl/acl_base.h"
#include "tools/error_handler/error_handler.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace tools {
namespace ascend {
bool ASCEND_RES_MANAGER_EXPORT NeedSaveAsyncCkpt();
bool ASCEND_RES_MANAGER_EXPORT NeedSaveSnapshot();

class ASCEND_RES_MANAGER_EXPORT AscendSnapshotMgr : public SnapshotMgr {
 public:
  static std::shared_ptr<AscendSnapshotMgr> GetInstance();

  AscendSnapshotMgr() = default;
  ~AscendSnapshotMgr();
  // disable copy constructor and the assignment operator
  AscendSnapshotMgr(const AscendSnapshotMgr &) = delete;
  AscendSnapshotMgr &operator=(const AscendSnapshotMgr &) = delete;

  void Clear();

  void RecordEvent(aclrtStream stream);
  void ResetEvent(aclrtStream stream);
  void StreamWaitEvent(aclrtStream stream);

  void SaveParameters(const std::vector<AnfNodePtr> &weights, aclrtStream stream);

 private:
  // async event for synchronization between compute stream and paramter's d2h copy stream
  aclrtEvent async_copy_event_ = nullptr;
};

using AscendSnapshotMgrPtr = std::shared_ptr<AscendSnapshotMgr>;
}  // namespace ascend
}  // namespace tools
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_ERROR_MANAGER_H
