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
#include "debug/data_dump/device_statistic/mean.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "debug/data_dump/device_statistic/kernel_factory.h"

namespace mindspore {
namespace datadump {

DeviceAddressPtr MeanStatisticKernel::GetAxisDeviceAddress(size_t dim) {
  vector<int64_t> axes(dim);
  for (size_t i = 0; i < dim; i++) {
    axes[i] = static_cast<int64_t>(i);
  }
  ShapeVector axes_shape{static_cast<int64_t>(dim)};
  size_t axisbytes = UnitSizeInBytes(kNumberTypeInt64) * dim;
  return GenerateDeviceAddress(axisbytes, kNumberTypeInt64, axes_shape, MakeValue(axes));
}

DeviceAddressPtr MeanStatisticKernel::GetKeepDimsDeviceAddress() {
  ShapeVector keepdims_shape = {};
  return GenerateDeviceAddress(UnitSizeInBytes(kNumberTypeBool), kNumberTypeBool, keepdims_shape, MakeValue(false));
}

DeviceAddressPtr MeanStatisticKernel::GetOutputDeviceAddress(const TypeId) {
  ShapeVector shape_vec = {};
  return GenerateDeviceAddress(UnitSizeInBytes(kNumberTypeFloat32), kNumberTypeFloat32, shape_vec);
}

DeviceAddressPtr MeanStatisticKernel::GetDtypeDeviceAddress(const TypeId type_id) {
  ShapeVector dtype_shape_vec = {1};
  return GenerateDeviceAddress(UnitSizeInBytes(type_id), type_id, dtype_shape_vec);
}

vector<DeviceAddressPtr> MeanStatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *input) {
  MS_EXCEPTION_IF_NULL(input);
  vector<DeviceAddressPtr> extra_inputs;

  auto axis = GetAxisDeviceAddress(input->GetShapeVector().size());
  MS_EXCEPTION_IF_NULL(axis);
  extra_inputs.emplace_back(axis);

  auto keepdims = GetKeepDimsDeviceAddress();
  MS_EXCEPTION_IF_NULL(keepdims);
  extra_inputs.emplace_back(keepdims);

  auto dtype = GetDtypeDeviceAddress(kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(dtype);
  extra_inputs.emplace_back(dtype);
  return extra_inputs;
}

REGISTER_KERNEL(KStatMean, MeanStatisticKernel);

}  // namespace datadump
}  // namespace mindspore
