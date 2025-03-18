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

#include "kernel/ascend/pyboost/customize/inplace_index_fill_tensor.h"
#include <memory>
#include <string>
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

ScalarPtr ConvertTensorToScalar(const std::string &prim_name, const BaseTensorPtr &value) {
  bool is_host_tensor = value->device_address() == nullptr && value->isa<Tensor>();
  if (is_host_tensor) {
    return CreateValueFromTensor(value->cast<TensorPtr>())->cast<ScalarPtr>();
  }

  ScalarPtr value_scalar = nullptr;
  value->data_sync();
  TypeId value_type_id = static_cast<TypeId>(value->data_type_c());
  switch (value_type_id) {
    case kNumberTypeBool: {
      auto value_num = static_cast<bool *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeBFloat16: {
      auto value_num = static_cast<bfloat16 *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeFloat16: {
      auto value_num = static_cast<float16 *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeFloat32: {
      auto value_num = static_cast<float *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeFloat64: {
      auto value_num = static_cast<double *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeInt8: {
      auto value_num = static_cast<int8_t *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeInt16: {
      auto value_num = static_cast<int16_t *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeInt32: {
      auto value_num = static_cast<int32_t *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    case kNumberTypeInt64: {
      auto value_num = static_cast<int64_t *>(value->data_c());
      MAKE_SCALAR((*value_num), value_type_id, value_scalar);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "For [" << prim_name << "], the input 'value'"
                        << " only supports Bool, BFloat16, Float16, Float32, Float64, Int8, Int16, Int32 and Int64,"
                        << " but got " << TypeIdToString(value_type_id);
  }
  return value_scalar;
}

tensor::BaseTensorPtr InplaceIndexFillTensorAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                            const BaseTensorPtr &input,
                                                            const Int64ImmPtr &dim,
                                                            const BaseTensorPtr &index,
                                                            const BaseTensorPtr &value) {
  auto index_shape = index->shape();
  auto value_shape = value->shape();
  if (MS_UNLIKELY(index_shape.size() > 1)) {
    MS_LOG(EXCEPTION) << "For [" << op->primitive()->name() << "], the rank of input 'index'"
                      << " must be in [0, 1], but got " << index_shape.size() << ".";
  }
  if (MS_UNLIKELY(value_shape.size() != 0)) {
    MS_LOG(EXCEPTION) << "For [" << op->primitive()->name() << "], the rank of input 'value'"
                      << " must be equal 0, but got " << value_shape.size() << ".";
  }
  auto value_scalar = ConvertTensorToScalar(op->primitive()->name(), value);
  MS_EXCEPTION_IF_NULL(value_scalar);
  auto dim_imm = GetValue<int64_t>(dim);
  std::vector<int64_t> index_vector;
  index->data_sync();
  TypeId index_type_id = static_cast<TypeId>(index->data_type_c());
  size_t elem_num = index->DataSize();
  if (index_type_id == TypeId::kNumberTypeInt64) {
    int64_t *elem_ptr = static_cast<int64_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (index_type_id == TypeId::kNumberTypeInt32) {
    int32_t *elem_ptr = static_cast<int32_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For [" << op->primitive()->name() << "], the input 'index'"
                            << " for conversion to int array must be of type Int32 or Int64,"
                            << " but got " << TypeIdToString(index_type_id);
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  op->set_outputs({input});

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input, dim_imm, index_vector, value_scalar]() {
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input);
      LAUNCH_ACLNN(aclnnInplaceIndexFillTensor, device_context, op->stream_id(), input, dim_imm, index_vector,
                   value_scalar);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
