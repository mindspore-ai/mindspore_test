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

#include "kernel/ascend/acl_ir/custom/custom_op_api_exec.h"

namespace mindspore::device::ascend {
namespace {
void CheckParamsSize(size_t params_size, size_t input_output_types_size) {
  if (params_size < kSizeTwo) {
    MS_LOG(EXCEPTION) << "Params number is invalid, it should greater than 2, but got " << params_size;
  }
  if (params_size - kSizeTwo > input_output_types_size) {
    MS_LOG(EXCEPTION) << "Params size " << (params_size - 2) << " is greater than input_output_types_ "
                      << input_output_types_size;
  }
}
void ReleaseConvertTypesImpl(const std::vector<void *> &params,
                             const std::vector<CustomSupportType> &input_output_types) {
  CheckParamsSize(params.size(), input_output_types.size());
  for (size_t i = 0; i < params.size() - kIndex2; i++) {
    auto type = input_output_types[i];
    // The `float,int,bool,string`  type does not need to be released.
    // aclTensorList, aclScalar,aclArray need to be released. In the future, if support for the above types is needed,
    // remember to release them.
    if (type == CustomSupportType::kTypeTensor) {
      const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
      MS_EXCEPTION_IF_NULL(acl_tensor);
      Release(acl_tensor);
      break;
    }
  }
  Release(params[params.size() - kIndex2]);
  Release(params[params.size() - kIndex1]);
}
}  // namespace

std::vector<ShapeVector> CustomGraphCache::FillShapeListFromTuple(const std::vector<void *> &params) {
  CheckParamsSize(params.size(), input_output_types_.size());
  std::vector<ShapeVector> shape_list;
  for (size_t i = 0; i < params.size(); i++) {
    auto type = input_output_types_[i];
    if (type == CustomSupportType::kTypeTensor) {
      const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
      MS_EXCEPTION_IF_NULL(acl_tensor);
      GetShape(acl_tensor, &shape_list);
    } else {
      (void)shape_list.emplace_back(ShapeVector());
    }
  }
  return shape_list;
}

void CustomGraphCache::ReleaseConvertTypes(const std::vector<void *> &params) {
  ReleaseConvertTypesImpl(params, input_output_types_);
}

void CustomGraphCache::UpdateAddressForTensor(aclOpExecutor *executor,
                                              const std::vector<std::vector<void *>> &address_list,
                                              const std::vector<void *> &params) {
  CheckParamsSize(params.size(), input_output_types_.size());
  size_t valid_index = 0;
  for (size_t i = 0; i < params.size() - kIndex2; i++) {
    auto type = input_output_types_[i];
    // The `float,int,bool,string`  type does not need to be updated address.
    // aclTensorList, std::optional<aclTensor>, need to be updated address. In the future, if support for the above
    // types is needed, remember to update.
    if (type == CustomSupportType::kTypeTensor) {
      const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
      MS_EXCEPTION_IF_NULL(acl_tensor);
      UpdateAddress(executor, acl_tensor, address_list[i], &valid_index);
    }
  }
}

void CustomReleaseCall::ReleaseConvertTypes(std::vector<void *> params) {
  ReleaseConvertTypesImpl(params, input_output_types_);
}

}  // namespace mindspore::device::ascend
