
/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include <string>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace kernel {
internal::TensorFormat InternalKernelUtils::ToInternalFormat(Format format) {
  switch (format) {
    case FRACTAL_NZ:
      return internal::TensorFormat::TENSOR_FORMAT_FRACTAL_NZ;
    default:
      // some op not support NCHW, NHWC, ... format, current return ND format
      return internal::TensorFormat::TENSOR_FORMAT_ND;
  }
}

int InternalKernelUtils::ToInternalOpId(std::string name) {
  if (ms_op_key_to_internel_op_id.find(name) != ms_op_key_to_internel_op_id.end()) {
    return ms_op_key_to_internel_op_id[name];
  }
  return -1;
}

internal::TensorDType InternalKernelUtils::ToInternalDType(TypeId type) {
  switch (type) {
    // float data type
    case kNumberTypeFloat16:
      return internal::TensorDType::TENSOR_DTYPE_FLOAT16;
    case kNumberTypeBFloat16:
      return internal::TensorDType::TENSOR_DTYPE_BF16;
    case kNumberTypeFloat32:
      return internal::TensorDType::TENSOR_DTYPE_FLOAT;
    case kNumberTypeDouble:
      return internal::TensorDType::TENSOR_DTYPE_DOUBLE;

    // int data type
    case kNumberTypeInt32:
      return internal::TensorDType::TENSOR_DTYPE_INT32;
    case kNumberTypeUInt32:
      return internal::TensorDType::TENSOR_DTYPE_UINT32;
    case kNumberTypeInt16:
      return internal::TensorDType::TENSOR_DTYPE_INT16;
    case kNumberTypeUInt16:
      return internal::TensorDType::TENSOR_DTYPE_UINT16;
    case kNumberTypeInt8:
      return internal::TensorDType::TENSOR_DTYPE_INT8;
    case kNumberTypeUInt8:
      return internal::TensorDType::TENSOR_DTYPE_INT8;
    case kNumberTypeInt64:
      return internal::TensorDType::TENSOR_DTYPE_INT64;
    case kNumberTypeUInt64:
      return internal::TensorDType::TENSOR_DTYPE_UINT64;

    // complex data type
    case kNumberTypeComplex64:
      return internal::TensorDType::TENSOR_DTYPE_COMPLEX64;
    case kNumberTypeComplex128:
      return internal::TensorDType::TENSOR_DTYPE_COMPLEX128;

    // other data type
    case kNumberTypeBool:
      return internal::TensorDType::TENSOR_DTYPE_BOOL;
    case kObjectTypeString:
      return internal::TensorDType::TENSOR_DTYPE_STRING;
    default:
      return internal::TensorDType::TENSOR_DTYPE_UNDEFINED;
  }
}

void InternalKernelUtils::ToInternalTensor(internal::Tensor *internal_tensor, const KernelTensor *kernel_tensor) {
  internal_tensor->desc.format = ToInternalFormat(kernel_tensor->format());
  internal_tensor->desc.dtype = ToInternalDType(kernel_tensor->dtype_id());
  if (kernel_tensor->GetShapeVector().size() == kDim0) {
    internal_tensor->desc.dims = {kDim1};
  } else {
    internal_tensor->desc.dims = internal::VecToSVec<int64_t>(kernel_tensor->GetShapeVector());
  }

  internal_tensor->data = kernel_tensor->device_ptr();
}

internal::DeviceRawBuf InternalKernelUtils::ToDeviceRawBuf(const KernelTensor *kernel_tensor) {
  return internal::DeviceRawBuf{kernel_tensor->size(), kernel_tensor->device_ptr()};
}

inline void SplitStringToNum(const std::string &str, char delim, std::vector<int32_t> *output_list) {
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty() && std::all_of(item.begin(), item.end(), ::isdigit)) {
      (void)output_list->emplace_back(std::stoi(item));
    }
  }
}

void GetSeqLenFromGraphInputOrEnv(const std::string &kernel_name, const std::string &tensor_name,
                                  const std::string &env_name, std::vector<int32_t> *seq_len) {
  seq_len->clear();
  std::string seq_len_env = common::GetEnv(env_name);
  if (!seq_len_env.empty()) {
    // first use env value to set seq_len if exists
    SplitStringToNum(seq_len_env, ',', seq_len);
    MS_LOG(INFO) << "For op '" << kernel_name << "', set param seq_len with env '" << env_name << "' as " << (*seq_len);
    return;
  }
  auto &llm_manager = LLMManager::GetInstance();
  auto seq_length_tensor = llm_manager.get_graph_input(tensor_name);
  if (seq_length_tensor != nullptr) {
    // then use graph_input tensor value to set seq_len if saved
    auto seq_length_values = static_cast<int32_t *>(seq_length_tensor->data());
    auto seq_length_values_num = seq_length_tensor->nbytes() / sizeof(int32_t);
    for (size_t i = 0; i < seq_length_values_num; i++) {
      (*seq_len).emplace_back(seq_length_values[i]);
    }
    MS_LOG(INFO) << "For op '" << kernel_name << "', set param seq_len with graph_input '" << tensor_name << "' as "
                 << (*seq_len);
    return;
  }
  MS_LOG(INFO) << "For op '" << kernel_name << "', if custom op disabled, param seq_len must be set, but '"
               << tensor_name << "' is not in graph_input, and env '" << env_name << "' is not set";
}

std::vector<int32_t> ConvertActualSeqLengthsToVector(KernelTensor *const actual_seq_length_ptr) {
  MS_EXCEPTION_IF_NULL(actual_seq_length_ptr);
  std::vector<int32_t> actual_seq_lengths_vector;
  if (actual_seq_length_ptr->type_id() != kMetaTypeNone) {
    TypeId actual_seq_lengths_dtype_id = actual_seq_length_ptr->dtype_id();
    if (actual_seq_lengths_dtype_id == kNumberTypeInt64) {
      std::vector<int64_t> actual_seq_lengths_vector_64 =
        actual_seq_length_ptr->GetValueWithCheck<std::vector<int64_t>>();
      actual_seq_lengths_vector.assign(actual_seq_lengths_vector_64.begin(), actual_seq_lengths_vector_64.end());
    } else if (actual_seq_lengths_dtype_id == kNumberTypeInt32) {
      actual_seq_lengths_vector = actual_seq_length_ptr->GetValueWithCheck<std::vector<int32_t>>();
    } else {
      MS_LOG(EXCEPTION) << "actual_seq_lengths data type must be Int32 or Int64, but got "
                        << TypeIdToString(actual_seq_lengths_dtype_id);
    }
  }
  return actual_seq_lengths_vector;
}

}  // namespace kernel
}  // namespace mindspore
