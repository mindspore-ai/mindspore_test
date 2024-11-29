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
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include <thread>
#include <future>
#include "transform/acl_ir/op_api_util.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/pipeline/task/device_task.h"
#include "runtime/pynative/op_executor.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
int8_t GetCubeMathType(bool use_hf32) { return transform::OpApiUtil::GetCubeMathType(use_hf32); }
bool IsAllowMatmulHF32() { return transform::OpApiUtil::IsAllowMatmulHF32(); }
bool IsAllowConvHF32() { return transform::OpApiUtil::IsAllowConvHF32(); }

std::pair<int64_t, int64_t> UpdateGeneratorState(const tensor::BaseTensorPtr &seed, const tensor::BaseTensorPtr &offset,
                                                 int64_t step) {
  runtime::Pipeline::Get().WaitAll();
  auto seed_value = *static_cast<int64_t *>(seed->data_c());
  offset->set_device_address(nullptr);
  auto offset_ptr = static_cast<int64_t *>(offset->data_c());
  auto offset_value = *offset_ptr;
  *offset_ptr += step;
  return {seed_value, offset_value};
}

bool ConvNDBatchify(const ShapeVector &input_shape, const int64_t num_spatial_dims, const std::string &func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  auto origin_shape_dim = SizeToLong(input_shape.size());
  const auto is_batched = (origin_shape_dim == dim_count_batch);
  if (origin_shape_dim != dim_count_no_batch && !is_batched) {
    MS_LOG(EXCEPTION) << "Expected " << dim_count_no_batch << "D (unbatched) or " << dim_count_batch
                      << "D (batched) input to " << func_name << ", but got input of size: " << origin_shape_dim;
  }
  return is_batched;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
