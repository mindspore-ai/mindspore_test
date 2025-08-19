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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTensorNum1 = 1;
constexpr size_t kTensorNum2 = 2;
constexpr size_t kTensorNum3 = 3;
constexpr size_t kTensorNum4 = 4;
constexpr size_t kTensorNum5 = 5;
constexpr size_t kTensorNum6 = 6;
constexpr size_t kTensorNum7 = 7;
constexpr size_t kTensorNum8 = 8;
constexpr size_t kTensorNum9 = 9;
constexpr size_t kTensorNum10 = 10;
constexpr size_t kTensorNum11 = 11;
constexpr size_t kTensorNum12 = 12;
constexpr auto kAclnnPrefix = "aclnn";
constexpr auto kCustomAclop = "custom_aclop";
constexpr auto kRegOpName = "reg_op_name";
constexpr auto kPrefixLen = 5;

template <size_t arg_num>
std::shared_ptr<AclnnKernelMod> CreateCustomAclnnKernelModInstance(const std::string &op_type) {
  return std::make_shared<custom::CustomAclnnKernelMod<arg_num>>(op_type);
}

template <size_t arg_num>
std::shared_ptr<pyboost::CustomAclnnPyboostKernelModBase> CreateCustomAclnnPyboostKernelModInstance(
  const std::string &op_type) {
  return std::make_shared<pyboost::CustomAclnnPyboostKernelMod<arg_num>>(op_type);
}

std::map<size_t, std::function<std::shared_ptr<AclnnKernelMod>(const std::string &)>> custom_aclnn_creators = {
  {kTensorNum1, &CreateCustomAclnnKernelModInstance<kTensorNum1>},
  {kTensorNum2, &CreateCustomAclnnKernelModInstance<kTensorNum2>},
  {kTensorNum3, &CreateCustomAclnnKernelModInstance<kTensorNum3>},
  {kTensorNum4, &CreateCustomAclnnKernelModInstance<kTensorNum4>},
  {kTensorNum5, &CreateCustomAclnnKernelModInstance<kTensorNum5>},
  {kTensorNum6, &CreateCustomAclnnKernelModInstance<kTensorNum6>},
  {kTensorNum7, &CreateCustomAclnnKernelModInstance<kTensorNum7>},
  {kTensorNum8, &CreateCustomAclnnKernelModInstance<kTensorNum8>},
  {kTensorNum9, &CreateCustomAclnnKernelModInstance<kTensorNum9>},
  {kTensorNum10, &CreateCustomAclnnKernelModInstance<kTensorNum10>},
  {kTensorNum11, &CreateCustomAclnnKernelModInstance<kTensorNum11>},
  {kTensorNum12, &CreateCustomAclnnKernelModInstance<kTensorNum12>}};

std::map<size_t, std::function<std::shared_ptr<CustomPyboostKernelMod>(const std::string &)>>
  custom_aclnn_pyboost_creators = {{kTensorNum1, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum1>},
                                   {kTensorNum2, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum2>},
                                   {kTensorNum3, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum3>},
                                   {kTensorNum4, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum4>},
                                   {kTensorNum5, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum5>},
                                   {kTensorNum6, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum6>},
                                   {kTensorNum7, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum7>},
                                   {kTensorNum8, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum8>},
                                   {kTensorNum9, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum9>},
                                   {kTensorNum10, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum10>},
                                   {kTensorNum11, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum11>},
                                   {kTensorNum12, &CreateCustomAclnnPyboostKernelModInstance<kTensorNum12>}};
}  // namespace

std::string AddPrefixForCustomNode(const std::string &op_type, bool unset) {
  MS_LOG(DEBUG) << "Start add prefix for " << op_type << ", unset is: " << unset;
  MS_VLOG(VL_CUSTOM_OP) << "Start add prefix for " << op_type << ", unset is: " << unset;
  if (unset) {
    return op_type;
  }
  if (op_type.length() >= kPrefixLen && op_type.substr(0, kPrefixLen) == kAclnnPrefix) {
    return op_type;
  }
  return kAclnnPrefix + op_type;
}

std::shared_ptr<AclnnKernelMod> GetCustomAclnnKernelMod(const std::string &op_type, size_t arg_num) {
  MS_LOG(INFO) << "Create custom aclnn kernel mod, op type : " << op_type << ", arg num : " << arg_num;
  auto it = custom_aclnn_creators.find(arg_num);
  if (it != custom_aclnn_creators.end()) {
    return it->second(op_type);
  } else {
    MS_LOG(ERROR) << "Aclnn custom only support arg nums between 1 and 12, but get: " << arg_num;
    return nullptr;
  }
}

std::shared_ptr<AclnnKernelMod> GetCustomAclnnKernelMod(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  op_type = AddPrefixForCustomNode(op_type, primitive->GetAttr(kCustomAclop) != nullptr);

  if (primitive->HasAttr(kCustomInputsType)) {
    return std::make_shared<custom::CustomV2AclnnKernelMod>(op_type);
  }

  auto arg_num = AnfUtils::GetInputTensorNum(anf_node) + AnfUtils::GetOutputTensorNum(anf_node);
  return GetCustomAclnnKernelMod(op_type, arg_num);
}

std::shared_ptr<CustomPyboostKernelMod> GetCustomAclnnPyboostKernelMod(const std::string &op_type, size_t arg_num) {
  MS_LOG(INFO) << "Create custom aclnn pyboost kernel mod, op type : " << op_type << ", arg num : " << arg_num;
  auto aclnn_op_type = AddPrefixForCustomNode(op_type);
  auto it = custom_aclnn_pyboost_creators.find(arg_num);
  if (it != custom_aclnn_pyboost_creators.end()) {
    return it->second(aclnn_op_type);
  } else {
    MS_LOG(ERROR) << "Aclnn custom only support arg nums between 1 and 12, but get: " << arg_num;
    return nullptr;
  }
}

}  // namespace kernel
}  // namespace mindspore
