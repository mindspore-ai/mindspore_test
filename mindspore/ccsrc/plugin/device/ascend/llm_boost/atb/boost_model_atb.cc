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

#include "plugin/device/ascend/llm_boost/atb/boost_model_atb.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "mindapi/base/type_id.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/llm_boost/atb/workspace.h"
#include "runtime/pynative/op_executor.h"
#include "utils/singleton.h"

// atb
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/base/context_factory.h"

namespace mindspore {
namespace kernel {

uint64_t GetNewModelId() {
  static uint64_t modelId = 0;
  uint64_t newModelId = modelId++;
  return newModelId;
}
void BoostModelATB::RunTask(std::string taskName, std::function<int()> task) {
  MS_LOG(INFO) << "run task: " << taskName;
  auto &executor = runtime::OpExecutor::GetInstance();
  auto ms_task = std::make_shared<runtime::PyBoostDeviceTask>(task);
  executor.PushOpRunTask(ms_task);
}

void *BoostModelATB::GetWorkSpace(uint64_t bufferSize) {
  MS_LOG(INFO) << "Get WorkSpace, buffsize: " << bufferSize;
  void *workspace = nullptr;
  if (bufferSize > 0) {
    workspace = Singleton<Workspace>::Instance().GetWorkspaceBuffer(bufferSize);
    if (workspace == nullptr) {
      MS_LOG(ERROR) << "Allocate workspace memory failed";
    }
  }
  return workspace;
}

void BoostModelATB::InitContext() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context_);
  device_name_ = device_context_->device_context_key().device_name_;
  device_id_ = device_context_->device_context_key().device_id_;
  stream_id_ = device_context_->device_res_manager_->DefaultStream();
}

BoostModelATB::BoostModelATB(const std::string ModelName) : modelName_(ModelName) {
  modelId_ = GetNewModelId();
  InitContext();
  auto stream = device_context_->device_res_manager_->GetStream(stream_id_);
  if (stream == nullptr) {
    stream = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  }
  MS_EXCEPTION_IF_NULL(stream);
  context_ = atb_speed::ContextFactory::GetAtbContext(stream);
  MS_LOG(INFO) << "BoostModel new modelName:" << modelName_ << ", modelId:" << modelId_;
}

BoostModelATB::~BoostModelATB() {
  model_.reset();
  context_.reset();
  atb_speed::ContextFactory::FreeAtbContext();
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
}

atb::Tensor BoostModelATB::CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc) {
  tensor::TensorPtr newMsTensor = CreateMsTensorFromTensorDesc(tensorDesc);
  msInternalTensors_.push_back(newMsTensor);
  return MSTensor2Tensor(newMsTensor);
}

int64_t BoostModelATB::Init(const std::string &param) {
  MS_LOG(INFO) << "BoostModel init start, modelName_:" << modelName_;
  model_ = atb_speed::ModelFactory::CreateInstance(modelName_, param);
  if (model_ != nullptr) {
    MS_LOG(INFO) << "Get model from the ModelFactory, " << modelName_
                 << ". If other models also want to be obtained from the ModelFactory, "
                 << "please register it and set `namespace` and `model class name`. "
                 << "Examples: REGISTER_MODEL(chatglm2_6b, ChatGlm2CommonModelFa). "
                 << "And then set `chatglm2_6b_ChatGlm2CommonModelFa` as input modelName_.";
  } else {
    MS_LOG(ERROR) << "Not support modelName: " << modelName_ << ", not found in ModelFactory.";
    return RET_FAILED;
  }

  auto getWorkspaceFunc = std::bind(&BoostModelATB::GetWorkSpace, this, std::placeholders::_1);
  auto createInternalTensorFromDescFunc =
    std::bind(&BoostModelATB::CreateInternalTensorFromDesc, this, std::placeholders::_1);

  auto runTaskFunc = std::bind(&BoostModelATB::RunTask, this, std::placeholders::_1, std::placeholders::_2);
  int64_t atbStatus = 0;
  atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, runTaskFunc);

  MS_LOG(INFO) << "BoostModel init success";
  return atbStatus;
}

int64_t BoostModelATB::SetWeight(const std::vector<tensor::TensorPtr> &msWeightTensors) {
  MS_LOG(INFO) << "BoostModel set weight start";
  std::vector<atb::Tensor> atWeightTensors;
  MSTensor2Tensor(msWeightTensors, atWeightTensors);
  MS_LOG(INFO) << "BoostModel set weight success";
  return model_->SetWeight(atWeightTensors);
}

int64_t BoostModelATB::SetKVCache(const std::vector<tensor::TensorPtr> &msKCacheTensors,
                                  const std::vector<tensor::TensorPtr> &msVCacheTensors) {
  MS_LOG(INFO) << "BoostModel set kvcache start";
  std::vector<atb::Tensor> kCacheTensors;
  std::vector<atb::Tensor> vCacheTensors;
  MSTensor2Tensor(msKCacheTensors, kCacheTensors);
  MSTensor2Tensor(msVCacheTensors, vCacheTensors);
  if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
    for (auto &kCacheTensor : kCacheTensors) {
      if (kCacheTensor.desc.format == ACL_FORMAT_NCHW) {
        kCacheTensor.desc.format = ACL_FORMAT_ND;
      }
    }
    for (auto &vCacheTensor : vCacheTensors) {
      if (vCacheTensor.desc.format == ACL_FORMAT_NCHW) {
        vCacheTensor.desc.format = ACL_FORMAT_ND;
      }
    }
  }
  int64_t atbStatus = model_->SetKVCache(kCacheTensors, vCacheTensors);
  MS_LOG(INFO) << "BoostModel set kvcache success";
  return atbStatus;
}

std::vector<tensor::TensorPtr> BoostModelATB::Forward(const std::vector<tensor::TensorPtr> &input,
                                                      const std::string &param) {
  MS_LOG(INFO) << "BoostModel forward start";
  msInternalTensors_.clear();
  std::vector<atb::Tensor> inTensors;
  MSTensor2Tensor(input, inTensors);
  if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
    for (auto &inTensor : inTensors) {
      if (inTensor.desc.format == ACL_FORMAT_NCHW) {
        inTensor.desc.format = ACL_FORMAT_ND;
      }
    }
  }

  std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
  for (size_t i = 0; i < inTensors.size(); ++i) {
    inTensorDescs.at(i) = inTensors.at(i).desc;
  }
  std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
  atb::Status st = model_->InferShape(inTensorDescs, outTensorDescs);

  if (st != 0) {
    MS_LOG(ERROR) << "BoostModel  infer shape fail, error code: " << st;
  }

  std::vector<tensor::TensorPtr> msOutTensors(outTensorDescs.size());
  for (size_t i = 0; i < msOutTensors.size(); ++i) {
    MS_LOG(INFO) << "ModelTorch outTensorDescs[" << i
                 << "]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i));
    atb_speed::Timer timer;
    msOutTensors.at(i) = CreateMsTensorFromTensorDesc(outTensorDescs.at(i));
    atb_speed::GetSingleton<atb_speed::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
    atb_speed::GetSingleton<atb_speed::Statistic>().mallocTorchTensorSize +=
      atb::Utils::GetTensorSize(outTensorDescs.at(i));
  }

  std::vector<atb::Tensor> outTensors;
  MSTensor2Tensor(msOutTensors, outTensors);
  if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
    for (auto &outTensor : outTensors) {
      if (outTensor.desc.format == ACL_FORMAT_NCHW) {
        outTensor.desc.format = ACL_FORMAT_ND;
      }
    }
  }

  int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
  if (atbStatus != atb::NO_ERROR) {
    std::vector<tensor::TensorPtr> msNullOutTensors;
    return msNullOutTensors;
  }
  MS_LOG(INFO) << "BoostModel forward end";
  return msOutTensors;
}

int64_t BoostModelATB::ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                                      const std::string &param) {
  int64_t atbStatus = model_->Execute(context_.get(), inTensors, outTensors, param);
  executeCount_++;
  return atbStatus;
}

atb::Tensor BoostModelATB::MSTensor2Tensor(const tensor::TensorPtr &msTensor) {
  MS_LOG(DEBUG) << "Convert MsTensor to AtbTensor";
  atb::Tensor tensor;
  auto device_sync = msTensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);

  if (device_address == nullptr) {
    device_address = device_context_->device_res_manager_->CreateDeviceAddress(
      nullptr, static_cast<size_t>(msTensor->data().nbytes()), msTensor->shape(), mindspore::Format::ND,
      msTensor->data_type(), device_name_, device_id_, stream_id_);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_from_persistent_mem(msTensor->is_parameter());
    msTensor->set_device_address(device_address);
  }

  if (device_address->GetMutablePtr() == nullptr) {
    if (!device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context_->device_context_key().device_id_
                        << ", alloc size: " << device_address->GetSize() << "B.";
    }
    if (!device_address->SyncHostToDevice(msTensor->shape(), device_address->GetSize(), device_address->type_id(),
                                          kOpFormat_ND, msTensor->data_ptr())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
    }
  }

  static std::map<TypeId, aclDataType> dtypeMap = {
    {kNumberTypeBool, ACL_BOOL},       {kNumberTypeUInt8, ACL_UINT8},   {kNumberTypeInt8, ACL_INT8},
    {kNumberTypeFloat16, ACL_FLOAT16}, {kNumberTypeFloat, ACL_FLOAT},   {kNumberTypeInt32, ACL_INT32},
    {kNumberTypeInt64, ACL_INT64},     {kNumberTypeBFloat16, ACL_BF16}, {kNumberTypeInt16, ACL_INT16},
  };

  tensor.deviceData = msTensor->device_address()->GetMutablePtr();
  tensor.desc.format = ACL_FORMAT_ND;

  if (tensor.deviceData != nullptr) {
    tensor.desc.shape.dimNum = msTensor->DataDim();
    for (int64_t i = 0; i < msTensor->DataDim(); i++) {
      tensor.desc.shape.dims[i] = msTensor->shape()[i];
    }
  }

  auto it = dtypeMap.find(msTensor->data_type());
  if (it != dtypeMap.end()) {
    tensor.desc.dtype = it->second;
  } else {
    MS_LOG(ERROR) << "not support dtype:" << msTensor->data_type();
  }
  tensor.dataSize = msTensor->data().nbytes();
  return tensor;
}

const tensor::TensorPtr BoostModelATB::CreateMsTensorFromTensorDesc(const atb::TensorDesc &tensorDesc) {
  static std::map<aclDataType, TypeId> dtypeMap = {
    {ACL_BOOL, kNumberTypeBool},       {ACL_UINT8, kNumberTypeUInt8},   {ACL_INT8, kNumberTypeInt8},
    {ACL_FLOAT16, kNumberTypeFloat16}, {ACL_FLOAT, kNumberTypeFloat},   {ACL_INT32, kNumberTypeInt32},
    {ACL_INT64, kNumberTypeInt64},     {ACL_BF16, kNumberTypeBFloat16},
  };

  TypeId msTensorType;
  ShapeVector msTensorShape;
  auto it = dtypeMap.find(tensorDesc.dtype);
  if (it != dtypeMap.end()) {
    msTensorType = it->second;
  } else {
    MS_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
  }

  for (uint64_t i = 0; i < tensorDesc.shape.dimNum; i++) {
    msTensorShape.push_back(tensorDesc.shape.dims[i]);
  }
  tensor::TensorDataPtr data = tensor::MakeTensorData(msTensorType, msTensorShape);
  tensor::TensorPtr msTensor = std::make_shared<tensor::Tensor>(msTensorType, msTensorShape, data);

  auto device_sync = msTensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  if (device_address == nullptr) {
    device_address = device_context_->device_res_manager_->CreateDeviceAddress(
      nullptr, static_cast<size_t>(msTensor->data().nbytes()), msTensor->shape(), mindspore::Format::ND,
      msTensor->data_type(), device_name_, device_id_, stream_id_);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_from_persistent_mem(msTensor->is_parameter());
    msTensor->set_device_address(device_address);
  }

  if (device_address->GetMutablePtr() == nullptr) {
    if (!device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context_->device_context_key().device_id_
                        << ", alloc size: " << device_address->GetSize() << "B.";
    }
  }
  return msTensor;
}

int64_t BoostModelATB::MSTensor2Tensor(const std::vector<tensor::TensorPtr> &msTensors,
                                       std::vector<atb::Tensor> &opsTensors) {
  for (auto &msTensor : msTensors) {
    atb::Tensor tensor = MSTensor2Tensor(msTensor);
    opsTensors.push_back(tensor);
  }
  return atb::NO_ERROR;
}

extern "C" BACKEND_EXPORT std::shared_ptr<BoostBaseModel> CreateAtbBoostModel(const std::string modelName) {
  return std::make_shared<BoostModelATB>(modelName);
}
}  // namespace kernel
}  // namespace mindspore
