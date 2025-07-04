/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hccl_kernel.h"

#include <map>
#include <set>
#include <unordered_set>
#include "mindspore/ops/op_def/ascend_op_name.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "mindspore/ops/op_def/array_op_name.h"
#include "mindspore/ops/op_def/framework_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/hccl_adapter/hccl_adapter.h"
#include "plugin/res_manager/ascend/collective/multi_ascend_collective_comm_lib.h"
#include "plugin/res_manager/ascend/collective/ascend_collective_comm_lib.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_memory_manager.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;
using MultiAscendCollectiveCommLib = mindspore::device::ascend::MultiAscendCollectiveCommLib;
namespace {
static std::map<std::string, std::string> kMsOpNameToHcomHcclType = {
  {mindspore::kAllReduceOpName, mindspore::kHcomOpTypeAllReduce},
  {mindspore::kReduceOpName, mindspore::kHcomOpTypeReduce},
  {mindspore::kCollectiveScatterOpName, mindspore::kHcomOpTypeScatter},
  {mindspore::kCollectiveGatherOpName, mindspore::kHcomOpTypeGather},
  {mindspore::kAllGatherOpName, mindspore::kHcomOpTypeAllGather},
  {mindspore::kBroadcastOpName, mindspore::kHcomOpTypeBroadcast},
  {mindspore::kSendOpName, mindspore::kHcomOpTypeSend},
  {mindspore::kReceiveOpName, mindspore::kHcomOpTypeReceive},
  {mindspore::kReduceScatterOpName, mindspore::kHcomOpTypeReduceScatter},
  {mindspore::kBarrierOpName, mindspore::kHcomOpTypeBarrier},
  {mindspore::kBatchISendIRecvOpName, mindspore::kHcomOpTypeBatchSendRecv},
  {mindspore::kAlltoAllVOpName, mindspore::kHcomOpTypeAlltoAllV},
};
std::string MsOpNameToHcomOpType(const std::string &ms_op_type) {
  auto iter = kMsOpNameToHcomHcclType.find(ms_op_type);
  if (iter == kMsOpNameToHcomHcclType.end()) {
    MS_LOG(EXCEPTION) << "Invalid MsOpType:" << ms_op_type;
  }
  return iter->second;
}
}  // namespace

namespace mindspore {
namespace kernel {
void CheckReduceOpUnderComplexInput(const std::vector<KernelTensor *> &inputs, const PrimitivePtr &prim,
                                    const HcclReduceOp &op_type) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!inputs.empty() && (*inputs.cbegin())->dtype_id() == TypeId::kNumberTypeComplex64 &&
      op_type != ::HcclReduceOp::HCCL_REDUCE_SUM) {
    std::string hcom_op_type;
    HcomUtil::GetHcomAttr<std::string>(prim, kAttrOp, &hcom_op_type);
    MS_LOG(EXCEPTION) << prim->name() << " doesn't support " << hcom_op_type
                      << " and just support sum in the case of complex input.";
  }
}

void HcclKernelFactory::Register(const std::string &name, HcclKernelCreater &&fun) {
  hccl_kernel_map_.emplace(name, fun);
}

std::shared_ptr<HcclKernel> HcclKernelFactory::Get(const std::string &name) {
  const auto &map = Get().hccl_kernel_map_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

HcclKernelFactory &HcclKernelFactory::Get() {
  static HcclKernelFactory _this{};
  return _this;
}

HcclKernel::HcclKernel()
    : hccl_count_(0),
      op_type_(::HcclReduceOp::HCCL_REDUCE_SUM),
      root_id_(0),
      src_rank_(0),
      dest_rank_(0),
      comm_(nullptr),
      use_lccl_{false} {}

bool IsSupportLccl(const std::string &group_name, const std::string &kernel_name,
                   const std::unordered_set<std::string> &lccl_enabled_groups) {
#ifdef ENABLE_INTERNAL_KERNELS
  bool enable_lccl = device::ascend::AscendHalManager::GetInstance().EnableLccl();
  std::set<std::string> support_lccl_op_names = {kAllReduceOpName, kReduceScatterOpName,   kBroadcastOpName,
                                                 kAllGatherOpName, kMatMulAllReduceOpName, kBarrierOpName};
  if (enable_lccl && lccl_enabled_groups.find(group_name) != lccl_enabled_groups.end() &&
      support_lccl_op_names.find(kernel_name) != support_lccl_op_names.end()) {
    return true;
  } else {
    return false;
  }
#endif
  return false;
}

bool HcclKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  // set source/destination rank
  if (kernel_name_ == kSendOpName || kernel_name_ == kReduceOpName || kernel_name_ == kMuxSendOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrDestRank, &dest_rank_)) {
      MS_LOG(ERROR) << "GetHcomDestRank fail!";
      return false;
    }
  } else if (kernel_name_ == kReceiveOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrSrcRank, &src_rank_)) {
      MS_LOG(ERROR) << "GetHcomSrcRank fail!";
      return false;
    }
  }

  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return false;
  }

  static std::set<std::string> reduce_op_names = {kAllReduceOpName, kReduceScatterOpName, kReduceOpName,
                                                  kReduceScatterVOpName, kMatMulAllReduceOpName};
  if (reduce_op_names.count(kernel_name_) != 0) {
    if (!HcomUtil::GetHcomOperationType(primitive_, &op_type_, &collective_reduce_type_)) {
      MS_LOG(ERROR) << "GetHcomOperationType fail!";
      return false;
    }
    CheckReduceOpUnderComplexInput(inputs, primitive_, op_type_);
  } else if (kernel_name_ == kBroadcastOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrRootRank, &root_id_)) {
      MS_LOG(ERROR) << "GetHcomRootId fail!";
      return false;
    }
  }

  if (!HcomUtil::GetHcomAttr<std::string>(primitive_, kAttrGroup, &group_)) {
    return false;
  }

  if (common::GetEnv(kSimulationLevel).empty()) {
    // Before calling each hccl operator, we need to wait for communicator to be initialized.
    distributed::collective::CollectiveManager::instance()->WaitCommInitDone(group_);
#ifdef ENABLE_INTERNAL_KERNELS
    std::unordered_set<std::string> lccl_enabled_groups =
      MultiAscendCollectiveCommLib::GetInstance().GetLcclEnabledGroups();
    use_lccl_ = IsSupportLccl(group_, kernel_name_, lccl_enabled_groups);
    if (use_lccl_) {
      LoadLcclLibrary();
    } else {
      LoadHcclLibrary();
    }
#else
    LoadHcclLibrary();
#endif
  }
  CalLoopSize();

  return true;
}

HcclDataType HcclKernel::GetHcclDataType() const {
  if (hccl_data_type_list_.empty()) {
    MS_LOG(EXCEPTION) << "list hccl_data_type_list_ is empty.";
  }
  return hccl_data_type_list_[0];
}

void HcclKernel::CalLoopSize() {
  int64_t rank_size = 1;
  int64_t fusion = 0;

  (void)HcomUtil::GetHcomAttr<int64_t>(primitive_, kAttrRankSize, &rank_size);
  (void)HcomUtil::GetHcomAttr<int64_t>(primitive_, kAttrFusion, &fusion);

  if (hccl_data_type_list_.size() != hccl_kernel_input_shape_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data type size " << hccl_data_type_list_.size() << " diff shape size "
                      << hccl_kernel_input_shape_list_.size();
  }
  loop_size_ = hccl_data_type_list_.size();
  if (hccl_kernel_input_shape_list_.size() > 1 && (kernel_name_ == kAllGatherOpName) && fusion >= 1) {
    loop_size_ *= static_cast<ulong>(rank_size);
  }
  if (kernel_name_ == kReduceScatterOpName && fusion >= 1) {
    loop_size_ = hccl_kernel_output_shape_list_.size();
  }
  if (kernel_name_ == kAlltoAllVOpName || kernel_name_ == kAllToAllOpName) {
    loop_size_ = hccl_kernel_output_shape_list_.size();
  }
  // For MatMulAllReduce, output number is 1.
  if (kernel_name_ == kMatMulAllReduceOpName) {
    loop_size_ = hccl_kernel_output_shape_list_.size();
  }
  MS_LOG(INFO) << "Get Hccl Kernel: " << kernel_name_ << ", output size: " << loop_size_;
}

bool HcclKernel::CalcTypeShapeAndCount(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  hccl_kernel_input_shape_list_.clear();
  hccl_kernel_output_shape_list_.clear();

  // set hccl kernel input/output shape
  std::function<ShapeVector(KernelTensor *)> GetTensorShape;
  if (!inputs.empty() && (*inputs.cbegin())->dtype_id() == TypeId::kNumberTypeComplex64) {
    GetTensorShape = [](KernelTensor *kernel_tensor) {
      // When the input type is Complex64, the type is converted to Float32 and the shape is increased
      auto re_shape = kernel_tensor->GetShapeVector();
      re_shape.push_back(kComplex64ConvertFloat32Num);
      return re_shape;
    };
  } else {
    GetTensorShape = [](KernelTensor *kernel_tensor) { return kernel_tensor->GetShapeVector(); };
  }

  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(hccl_kernel_input_shape_list_), GetTensorShape);
  std::transform(outputs.cbegin(), outputs.cend(), std::back_inserter(hccl_kernel_output_shape_list_), GetTensorShape);

  // set hccl data_type and count
  if (!HcomUtil::GetHcomDataType(kernel_name_, inputs, outputs, &hccl_data_type_list_)) {
    MS_LOG(ERROR) << "GetHcomDataType fail!";
    return false;
  }
  if (!HcomUtil::GetHcomCount(
        primitive_, hccl_data_type_list_,
        HcomUtil::IsReceiveOp(kernel_name_) ? hccl_kernel_output_shape_list_ : hccl_kernel_input_shape_list_,
        inputs.size(), std::nullopt, &hccl_count_)) {
    MS_LOG(ERROR) << "GetHcomCount fail!";
    return false;
  }

  return true;
}

bool HcclKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  if (inputs.empty() && outputs.empty()) {
    MS_LOG(ERROR) << "Hccl kernel input or output is empty.";
    return false;
  }
  if (hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Hccl data type list is empty.";
    return false;
  }

  MS_LOG(INFO) << "Start Execute: " << kernel_name_;
  std::string hccl_type = MsOpNameToHcomOpType(kernel_name_);
  HcclDataType data_type = hccl_data_type_list_[0];

  ::HcomOperation op_info;
  op_info.hcclType = hccl_type;
  op_info.inputPtr = inputs[0]->device_ptr();
  op_info.outputPtr = outputs[0]->device_ptr();
  op_info.dataType = static_cast<HcclDataType>(data_type);
  op_info.opType = static_cast<HcclReduceOp>(op_type_);
  op_info.root = root_id_;
  op_info.count = hccl_count_;

  auto callback = [this](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcomExcutorInitialize failed, ret:" << status;
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    MS_LOG(INFO) << "Hccl callback success.";
  };

  auto hccl_ret = hccl::HcclAdapter::GetInstance().HcclExecEnqueueOp(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call EnqueueHcomOperation failed, node info: " << kernel_name_;
  }

  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);
  MS_LOG(INFO) << "Execute " << kernel_name_ << " success.";
  return true;
}

int HcclKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }

  // update output_size_list_
  output_size_list_.clear();
  for (ulong i = 0; i < loop_size_; ++i) {
    size_t size = 0;
    if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
    }
    output_size_list_.push_back(size);
  }

  return KRET_OK;
}

void HcclKernel::LoadHcclLibrary() {
  comm_ = AscendCollectiveCommLib::GetInstance().HcclCommunicator(group_);
  if (common::UseHostCollective() && !hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    MS_EXCEPTION_IF_NULL(comm_);
    primitive_->set_attr(kAttrComm, MakeValue<int64_t>(reinterpret_cast<int64_t>(comm_)));
  }
  hccl_inner_comm_name_ = AscendCollectiveCommLib::GetInstance().CommName(group_);
  primitive_->set_attr(kAttrCollectiveCommLib, MakeValue<std::string>("HCCL"));
}

bool HcclKernel::NeedReGetHcom() { return mindspore::UCEException::GetInstance().rebuild_group_flag(); }

#ifdef ENABLE_INTERNAL_KERNELS
void HcclKernel::LoadLcclLibrary() {
  std::string lowlatency_comm_lib_name = "liblowlatency_collective.so";
  auto loader = std::make_shared<CollectiveCommLibLoader>(lowlatency_comm_lib_name);
  MS_EXCEPTION_IF_NULL(loader);
  if (!loader->Initialize()) {
    MS_LOG(EXCEPTION) << "Loading LCCL collective library failed.";
  }
  lowlatency_comm_lib_handle_ = loader->collective_comm_lib_ptr();
  MS_EXCEPTION_IF_NULL(lowlatency_comm_lib_handle_);

  auto get_lccl_func = DlsymFuncObj(LcclCommunicator, lowlatency_comm_lib_handle_);
  lccl_ptr_ = get_lccl_func(group_);
  MS_EXCEPTION_IF_NULL(lccl_ptr_);
  primitive_->set_attr(kAttrCollectiveCommLib, MakeValue<std::string>("LCCL"));
}
#endif
}  // namespace kernel
}  // namespace mindspore
