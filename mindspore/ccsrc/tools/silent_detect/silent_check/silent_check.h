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
#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_CHECK_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_CHECK_H_

#include <map>
#include <memory>
#include <string>
#include "include/backend/visible.h"
#include "ir/tensor.h"
#include "mindspore/core/include/ir/tensor.h"

namespace mindspore {
namespace silentcheck {
const char kAttrSilentCheckOpType[] = "silent_check_type";
enum SilentCheckOpType : int { kSilentCheckGradLastOp = 0, kSilentCheckGradCommOp = 1 };
using tensor::TensorPtr;

class BACKEND_COMMON_EXPORT SilentCheckerBase {
 public:
  static std::shared_ptr<SilentCheckerBase> GetInstance();

  static bool Register(const std::string &name, const std::shared_ptr<SilentCheckerBase> &instance);

  static void ClearAll();

  SilentCheckerBase() = default;

  virtual ~SilentCheckerBase() = default;

  virtual void Clear() {}

  virtual void ClearCheckObjects() {}

  virtual bool IsNpuAsdEnable() { return false; }

  virtual void SetBackProp(bool is_back_prop) {}

  virtual void DoSilentCheck(const std::string &op_name, const std::string &comm_group, const TensorPtr &input_grad) {}

  bool NeedInsertCheckForLastGrad();

  void SetPipelineStage(uint32_t pp_stage) { pp_stage_ = pp_stage; }

 protected:
  // pipeline parallel group name
  uint32_t pp_stage_;

  static std::map<std::string, std::shared_ptr<SilentCheckerBase>> &GetInstanceMap();
};

}  // namespace silentcheck
}  // namespace mindspore

#define SILENT_CHECK_REG(NAME, CLAZZ)               \
  static bool g_SilentChecker_##NAME##_reg_result = \
    mindspore::silentcheck::SilentCheckerBase::Register(NAME, std::make_shared<CLAZZ>())

#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_CHECK_H_
