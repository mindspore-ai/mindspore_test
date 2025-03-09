/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORDUMP_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORDUMP_UTILS_H_

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <memory>
#include <utility>
#include <condition_variable>
#include "plugin/res_manager/ascend/mbuf_manager/mbuf_receive_manager.h"
#include "plugin/res_manager/ascend/visible.h"

namespace mindspore::device::ascend {
const std::pair<string, string> tensordump_mapping{"ms_tensor_dump", "TensorDump"};

class ASCEND_RES_MANAGER_EXPORT TensorDumpUtils {
 public:
  static TensorDumpUtils &GetInstance();

  TensorDumpUtils() = default;
  TensorDumpUtils(const TensorDumpUtils &) = delete;
  TensorDumpUtils &operator=(const TensorDumpUtils &) = delete;
  void SaveDatasetToNpyFile(const ScopeAclTdtDataset &dataset);

 private:
  std::string TensorNameToArrayName(std::string tensor_name, const std::string &data_type);
};

}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_TENSORDUMP_UTILS_H_
