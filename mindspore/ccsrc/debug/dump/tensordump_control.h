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
#ifndef MINDSPORE_CCSRC_DEBUG_TENSORDUMP_CONTROL_H_
#define MINDSPORE_CCSRC_DEBUG_TENSORDUMP_CONTROL_H_

#include <set>
#include <string>
#include <vector>
#include <array>
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/common/visible.h"

namespace mindspore {

class DUMP_EXPORT TensorDumpStepManager {
 public:
  static TensorDumpStepManager &GetInstance() {
    static TensorDumpStepManager instance;
    return instance;
  }
  ~TensorDumpStepManager() = default;
  void SetDumpStep(const std::vector<size_t> &);
  std::string ProcessFileName(const std::string &, const std::string &, const int = kGraphMode);
  void SetAclDumpCallbackReg(void *);

 private:
  TensorDumpStepManager() = default;
  DISABLE_COPY_AND_ASSIGN(TensorDumpStepManager);
  void UpdateStep(const int);
  size_t GetStep(const int) const;
  bool NeedDump(const int) const;
  std::string TensorNameToArrayName(std::string, std::string, const int);
  size_t FetchAddID();
  std::atomic<size_t> id_;
  std::array<size_t, 2> step_ = {0, 0};
  std::set<size_t> valid_steps_;
  void *aclDumpCallbackReg_;
};

}  // namespace mindspore

#endif
