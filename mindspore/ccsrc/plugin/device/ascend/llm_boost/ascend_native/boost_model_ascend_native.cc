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
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "plugin/device/ascend/llm_boost/ascend_native/boost_model_ascend_native.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "common/kernel.h"
#include "include/backend/device_address.h"
#include "include/llm/llama_impl.h"

namespace mindspore {
namespace kernel {
namespace {
bool IsDeviceTensor(mindspore::tensor::TensorPtr tensor) {
  auto addr = tensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(addr);
  if ((device_address != nullptr) && (true || device_address->GetDeviceType() != device::DeviceType::kCPU)) {
    return true;
  }
  return false;
}

static internal::TensorDType ToInternalDType(TypeId type) {
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

internal::Tensor *CreateInternalTensor(mindspore::tensor::TensorPtr tensor) {
  auto btensor = new internal::Tensor();
  btensor->desc.dtype = ToInternalDType(tensor->data_type());
  for (auto &val : tensor->shape_c()) {
    btensor->desc.dims.push_back(val);
  }
  if (IsDeviceTensor(tensor)) {
    btensor->data = tensor->data_c();
  } else {
    btensor->hostData = tensor->data_c();
  }
  btensor->dataSize = tensor->Size();
  return btensor;
}

TypeId ToMSDType(internal::TensorDType dtype) {
  switch (dtype) {
    // float data type
    case internal::TensorDType::TENSOR_DTYPE_FLOAT16:
      return kNumberTypeFloat16;
    case internal::TensorDType::TENSOR_DTYPE_BF16:
      return kNumberTypeBFloat16;
    case internal::TensorDType::TENSOR_DTYPE_FLOAT:
      return kNumberTypeFloat32;
    case internal::TensorDType::TENSOR_DTYPE_DOUBLE:
      return kNumberTypeDouble;
    // int data type
    case internal::TensorDType::TENSOR_DTYPE_INT32:
      return kNumberTypeInt32;
    case internal::TensorDType::TENSOR_DTYPE_UINT32:
      return kNumberTypeUInt32;
    case internal::TensorDType::TENSOR_DTYPE_INT16:
      return kNumberTypeInt16;
    case internal::TensorDType::TENSOR_DTYPE_UINT16:
      return kNumberTypeUInt16;
    case internal::TensorDType::TENSOR_DTYPE_INT8:
      return kNumberTypeInt8;
    case internal::TensorDType::TENSOR_DTYPE_UINT8:
      return kNumberTypeUInt8;
    case internal::TensorDType::TENSOR_DTYPE_INT64:
      return kNumberTypeInt64;
    case internal::TensorDType::TENSOR_DTYPE_UINT64:
      return kNumberTypeUInt64;
    // complex data type
    case internal::TensorDType::TENSOR_DTYPE_COMPLEX64:
      return kNumberTypeComplex64;
    case internal::TensorDType::TENSOR_DTYPE_COMPLEX128:
      return kNumberTypeComplex128;
      // other data type
    case internal::TensorDType::TENSOR_DTYPE_BOOL:
      return kNumberTypeBool;
    case internal::TensorDType::TENSOR_DTYPE_STRING:
      return kObjectTypeString;
    default:
      return kTypeUnknown;
  }
}

mindspore::tensor::TensorPtr CreateMSTensor(internal::Tensor *tensor) {
  TypeId data_type = ToMSDType(tensor->desc.dtype);
  ShapeVector shape;
  copy(tensor->desc.dims.begin(), tensor->desc.dims.end(), back_inserter(shape));
  void *data = tensor->hostData;
  size_t data_len = tensor->dataSize;
  auto btensor = std::make_shared<mindspore::tensor::Tensor>(data_type, shape, data, data_len);
  return btensor;
}
}  // namespace

int64_t BoostModelAscendC::InitData(const llm_data &data) {
  auto param = std::make_shared<internal::OpLlamaModelParam>();
  param->hcom_ = NULL;
  param->batch_size_ = data.batch_size;
  param->seq_length_ = data.seq_length;
  param->head_num_ = data.num_heads;
  param->kv_head_num_ = data.kv_head_num;
  param->hidden_size_ = data.hidden_size;
  param->num_layers_ = data.num_layers;
  param->ln_eps_ = data.rms_norm_eps;
  param->vocab_size_ = data.vocab_size;
  param->multiple_of_ = data.multiple_of;
  param->paged_attention_ = (data.page_num > 0);
  param->page_num_ = data.page_num;
  param->page_size_ = data.page_size;
  llama_ = std::make_shared<internal::LlamaImpl>(param);

  return KRET_OK;
}

int64_t BoostModelAscendC::Init(const std::string &param) { return KRET_OK; }

std::vector<tensor::TensorPtr> BoostModelAscendC::Forward(const std::vector<tensor::TensorPtr> &vec,
                                                          const std::string &param) {
  auto llama = std::static_pointer_cast<internal::LlamaImpl>(llama_);
  std::vector<mindspore::tensor::TensorPtr> outvec;
  auto &invec = llama->get_inputs();
  if (invec.size() != vec.size()) {
    MS_LOG(ERROR) << "input size do not match model:" << invec.size() << " actual:" << vec.size() << std::endl;
  }
  for (size_t i = 0; i < vec.size(); i++) {
    auto &tensor = vec.at(i);
    auto *data = tensor->data_c();
    auto &btensor = invec.at(i);
    if (IsDeviceTensor(tensor)) {
      btensor->data = data;
    } else {
      btensor->hostData = data;
    }
    btensor->desc.dims.clear();
    for (auto &val : tensor->shape_c()) {
      btensor->desc.dims.push_back(val);
    }
  }
  llama->Launch();
  auto &outputs = llama->get_outputs();
  for (auto &t : outputs) {
    auto tensor = CreateMSTensor(t);
    outvec.push_back(tensor);
  }
  return outvec;
}

int64_t BoostModelAscendC::SetWeightMap(const std::map<std::string, mindspore::tensor::TensorPtr> &map) {
  auto llama = std::static_pointer_cast<internal::LlamaImpl>(llama_);
  auto converted_map = new std::map<std::string, internal::Tensor *>();
  for (auto &item : map) {
    auto &name = item.first;
    auto tensor = item.second;
    auto btensor = CreateInternalTensor(tensor);
    (*converted_map)[name] = btensor;
  }
  llama->SetupWeights(converted_map);
  llama->AclInit();
  llama->Init();
  return KRET_OK;
}

int64_t BoostModelAscendC::AddFlags(bool is_first) {
  auto llama = std::static_pointer_cast<internal::LlamaImpl>(llama_);
  if (llama) {
    llama->SetIsFIrstIter(is_first);
  }
  return KRET_OK;
}

extern "C" BACKEND_EXPORT std::shared_ptr<BoostBaseModel> CreateAscendNativeBoostModel(const std::string modelName) {
  return std::make_shared<BoostModelAscendC>(modelName);
}
}  // namespace kernel
}  // namespace mindspore
