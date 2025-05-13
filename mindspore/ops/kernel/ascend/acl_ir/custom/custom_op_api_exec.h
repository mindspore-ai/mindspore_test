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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_EXEC_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_EXEC_H_
#include <vector>
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "kernel/ascend/acl_ir/custom/custom_aclnn_utils.h"

namespace mindspore::device::ascend {
using CustomSupportType = mindspore::kernel::custom::CustomSupportType;
class CustomGraphCache {
 public:
  explicit CustomGraphCache(device::ascend::aclOpExecutor *executor, std::vector<void *> &&param,
                            std::vector<CustomSupportType> input_output_types)
      : executor_(executor), converted_params_(param), input_output_types_(input_output_types) {}
  std::vector<ShapeVector> operator()(const device::ascend::ProcessCacheType &process_cache_type,
                                      const std::vector<std::vector<void *>> &address_list = {}) {
    static auto release_executor_func = device::ascend::OpApiDefaultResource::GetInstance().release_executor_func();
    switch (process_cache_type) {
      case ProcessCacheType::kGetOutputShape:
        return FillShapeListFromTuple(converted_params_);
      case ProcessCacheType::kReleaseParamsAndExecutor:
        ReleaseConvertTypes(converted_params_);
        if (release_executor_func != nullptr) {
          release_executor_func(executor_);
        }
        break;
      case ProcessCacheType::kReleaseParams:
        ReleaseConvertTypes(converted_params_);
        break;
      case ProcessCacheType::kUpdateTensorAddress:
        UpdateAddressForTensor(executor_, address_list, converted_params_);
        break;
      default:
        break;
    }
    return {};
  }

 private:
  std::vector<ShapeVector> FillShapeListFromTuple(const std::vector<void *> &params);
  void ReleaseConvertTypes(const std::vector<void *> &params);
  void UpdateAddressForTensor(aclOpExecutor *executor, const std::vector<std::vector<void *>> &address_list,
                              const std::vector<void *> &params);
  device::ascend::aclOpExecutor *executor_;
  std::vector<void *> converted_params_;
  std::vector<CustomSupportType> input_output_types_;
};

class CustomReleaseCall {
 public:
  explicit CustomReleaseCall(std::vector<void *> &&param, std::vector<CustomSupportType> input_output_types)
      : converted_params_(param), input_output_types_(input_output_types) {}
  void operator()() {
    ReleaseConvertTypes(converted_params_);
    auto release_mem_func = device::ascend::OpApiDefaultResource::GetInstance().release_mem_func();
    if (release_mem_func) {
      release_mem_func(nullptr, false);
    }
  }

 private:
  void ReleaseConvertTypes(std::vector<void *> params);
  std::vector<void *> converted_params_;
  std::vector<CustomSupportType> input_output_types_;
};

}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_OP_API_EXEC_H_
