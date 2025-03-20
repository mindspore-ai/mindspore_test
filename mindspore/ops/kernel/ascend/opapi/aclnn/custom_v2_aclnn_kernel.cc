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

#include "kernel/ascend/opapi/aclnn/custom_v2_aclnn_kernel.h"
#include <algorithm>
#include <tuple>
#include "kernel/ascend/acl_ir/custom/custom_op_api_exec.h"
#include "kernel/ascend/acl_ir/custom/custom_op_api_cache.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
namespace custom {
namespace {
constexpr auto kCustomInputsType = "custom_inputs_type";
}

void CustomV2AclnnKernelMod::ConvertTypes(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs,
                                          std::vector<void *> *convert_inputs, std::vector<void *> *convert_outputs) {
  MS_EXCEPTION_IF_NULL(convert_inputs);
  MS_EXCEPTION_IF_NULL(convert_outputs);
  if (inputs.size() > input_output_types_.size()) {
    MS_LOG(EXCEPTION) << "Inputs size " << inputs.size()
                      << " is greater than input_output_types_ size: " << input_output_types_.size();
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto type = input_output_types_[i];
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        convert_inputs->emplace_back(device::ascend::ConvertType(inputs[i]));
        break;
      }
      case CustomSupportType::kTypeBool: {
        (void)inputs_bool_value_.emplace_back(inputs[i]->GetValueWithCheck<bool>());
        convert_inputs->emplace_back(&(inputs_bool_value_.back()));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        (void)inputs_float_value_.emplace_back(inputs[i]->GetValueWithCheck<float>());
        convert_inputs->emplace_back(&(inputs_float_value_.back()));
        break;
      }
      case CustomSupportType::kTypeInt: {
        (void)inputs_int_value_.emplace_back(inputs[i]->GetValueWithCheck<int64_t>());
        convert_inputs->emplace_back(&inputs_int_value_.back());
        break;
      }
      case CustomSupportType::kTypeString: {
        convert_inputs->emplace_back(const_cast<void *>(
          static_cast<const void *>(device::ascend::ConvertType(inputs[i]->GetValueWithCheck<std::string>()))));
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << type;
    }
  }

  std::transform(outputs.begin(), outputs.end(), std::back_inserter(*convert_outputs),
                 [](const auto &item) { return device::ascend::ConvertType(item); });
}

bool CustomV2AclnnKernelMod::CallGetWorkSpaceSize(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs,
                                                  uint64_t *workspace_size_addr, aclOpExecutor **executor_addr,
                                                  void *get_workspace_size_func) {
  std::vector<void *> convert_inputs;
  std::vector<void *> convert_outputs;
  ConvertTypes(inputs, outputs, &convert_inputs, &convert_outputs);
  converted_params_.clear();
  std::copy(convert_inputs.begin(), convert_inputs.end(), std::back_inserter(converted_params_));
  std::copy(convert_outputs.begin(), convert_outputs.end(), std::back_inserter(converted_params_));
  converted_params_.emplace_back(workspace_size_addr);
  converted_params_.emplace_back(executor_addr);

  std::string file_path;
  std::string func_type;
  const auto &exec_info = GetValue<std::string>(primitive_->GetAttr("func_name"));
  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For custom '" << op_type_ << "', couldn't find the AOT binary file under path: " << path;
    }
    file_path = real_path.value();
    func_type = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "For custom'" << op_type_ << "', user defined function path '" << exec_info << "' is illegal.";
  }

  auto handle_ = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    MS_LOG(EXCEPTION) << "For custom '" << op_type_ << ", dlopen file '" << file_path
                      << "' should be successful, but error occurs! Error message is: " << dlerror();
  }

  using FuncType = int (*)(void *, std::vector<void *>, std::vector<void *>, uint64_t *, aclOpExecutor **);
  auto func_name = func_type + "GetWorkSpaceSize";
  auto user_func = reinterpret_cast<FuncType>(dlsym(handle_, func_name.c_str()));
  if (user_func != nullptr) {
    int ret = 0;
    try {
      ret = user_func(get_workspace_size_func, convert_inputs, convert_outputs, workspace_size_addr, executor_addr);
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "For custom '" << op_type_ << "', operator failed when executing user defined func " << func_name
                    << "! "
                    << "Error message is " << e.what();
      return false;
    }
    return ret;
  } else {
    MS_LOG(EXCEPTION) << "For custom " << op_type_ << ", can not find func " << func_name << " in " << file_path;
  }
}

CacheTuple CustomV2AclnnKernelMod::GenCustomExecutorForResize(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  auto workspace_api_name = op_type_ + "GetWorkspaceSize";
  const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  device::ascend::aclOpExecutor *executor = nullptr;
  uint64_t *workspace_size_addr = &workspace_size;
  device::ascend::aclOpExecutor **executor_addr = &executor;
  auto workspace_status =
    CallGetWorkSpaceSize(inputs, outputs, workspace_size_addr, executor_addr, get_workspace_size_func_ptr);
  if (workspace_status != 0) {
    CHECK_AND_THROW_UCE_ERROR(workspace_api_name);
    MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";
  }

  device::ascend::SetExecutorRepeatable(workspace_api_name, executor);
  int32_t repeat_ret = device::ascend::SetExecutorRepeatable(workspace_api_name, executor);
  auto graph_cache = device::ascend::CustomGraphCache(executor, std::move(converted_params_), input_output_types_);
  auto process_cache = device::ascend::ProcessCache(graph_cache);
  return std::make_tuple(workspace_size, executor, process_cache, repeat_ret);
}

void CustomV2AclnnKernelMod::GetWorkspaceForResize(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  hash_id_ = device::ascend::CustomAclnnHash(op_type_, inputs, outputs, input_output_types_);
  size_t cur_workspace = 0;
  if (hash_map_.count(hash_id_)) {
    hash_cache_.splice(hash_cache_.begin(), hash_cache_, hash_map_[hash_id_]);
    cur_workspace = std::get<kWorkspaceIndex>(hash_cache_.front());
  } else {
    auto [workspace, executor, cache, fail_cache] = GenCustomExecutorForResize(inputs, outputs);
    cur_workspace = workspace;
    if (!fail_cache) {
      hash_cache_.emplace_front(hash_id_, executor, cache, workspace);
      hash_map_[hash_id_] = hash_cache_.begin();
    } else {
      hash_id_ = 0;
      cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
    }
  }
  if (hash_cache_.size() > capacity_) {
    hash_map_.erase(std::get<0>(hash_cache_.back()));
    auto release_func = std::get<kReleaseFuncIndex>(hash_cache_.back());
    release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
    hash_cache_.pop_back();
  }

  if (cur_workspace != 0) {
    std::vector<size_t> workspace_size_list = {cur_workspace};
    SetWorkspaceSizeList(workspace_size_list);
  }
}

void CustomV2AclnnKernelMod::RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto [executor, release_func] = GetExecutor(inputs, outputs);
  if (workspace_size_list_.empty()) {
    RUN_CUSTOM_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    auto workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_CUSTOM_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr,
                            release_func);
  }
}

std::vector<std::vector<void *>> CustomV2AclnnKernelMod::GetTensorAddress(const std::vector<KernelTensor *> &inputs,
                                                                          const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() > input_output_types_.size()) {
    MS_LOG(EXCEPTION) << "Inputs size " << inputs.size()
                      << " is greater than input_output_types_ size: " << input_output_types_.size();
  }
  std::vector<std::vector<void *>> address_list;
  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto type = input_output_types_[i];
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        address_list.emplace_back(device::ascend::GetAddr(inputs[i]));
        break;
      }
      case CustomSupportType::kTypeBool: {
        address_list.emplace_back(device::ascend::GetAddr(inputs[i]->GetValueWithCheck<bool>()));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        address_list.emplace_back(device::ascend::GetAddr(inputs[i]->GetValueWithCheck<float>()));
        break;
      }
      case CustomSupportType::kTypeInt: {
        address_list.emplace_back(device::ascend::GetAddr(inputs[i]->GetValueWithCheck<int64_t>()));
        break;
      }
      case CustomSupportType::kTypeString: {
        address_list.emplace_back(device::ascend::GetAddr(inputs[i]->GetValueWithCheck<std::string>()));
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << type;
    }
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    address_list.emplace_back((device::ascend::GetAddr(outputs[i])));
  }
  return address_list;
}

void CustomV2AclnnKernelMod::UpdateTensorForLaunch(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs,
                                                   const ProcessCache &cache) {
  const auto &address_list = GetTensorAddress(inputs, outputs);
  cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);
}

ExecutorTuple CustomV2AclnnKernelMod::GenCustomExecutor(const std::vector<KernelTensor *> &inputs,
                                                        const std::vector<KernelTensor *> &outputs) {
  auto workspace_api_name = op_type_ + "GetWorkspaceSize";
  static device::ascend::ApiCachePool api_cache_pool;
  const char *api_name = api_cache_pool.get(op_type_);
  const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  device::ascend::aclOpExecutor *executor = nullptr;
  std::function<void()> release_func = nullptr;
  uint64_t *workspace_size_addr = &workspace_size;
  device::ascend::aclOpExecutor **executor_addr = &executor;
  if (CustomHitCache(api_name, executor_addr, workspace_size_addr, inputs, outputs, input_output_types_)) {
    MS_LOG(DEBUG) << "gen executor aclnn cache hit.";
    return std::make_tuple(workspace_size, executor, release_func);
  }
  MS_LOG(DEBUG) << "gen executor aclnn cache miss.";
  auto init_mem_func = device::ascend::OpApiDefaultResource::GetInstance().init_mem_func();
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }

  auto workspace_status =
    CallGetWorkSpaceSize(inputs, outputs, workspace_size_addr, executor_addr, get_workspace_size_func_ptr);
  if (workspace_status != 0) {
    MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";
  }
  auto releas_call = device::ascend::CustomReleaseCall(std::move(converted_params_), input_output_types_);
  release_func = std::function<void()>(releas_call);
  auto uninit_mem_func = device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();
  if (uninit_mem_func) {
    uninit_mem_func(nullptr, false);
  }
  device::ascend::UninitCacheThreadLocal();
  return std::make_tuple(workspace_size, executor, release_func);
}

std::pair<aclOpExecutor *, std::function<void()>> CustomV2AclnnKernelMod::GetExecutor(
  const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (hash_id_ == 0 || !hash_map_.count(hash_id_)) {
    aclOpExecutor *executor;
    std::function<void()> release_func;
    std::tie(std::ignore, executor, release_func) = GenCustomExecutor(inputs, outputs);
    return std::make_pair(executor, release_func);
  }
  const auto &cur_run = *hash_map_[hash_id_];
  UpdateTensorForLaunch(inputs, outputs, std::get<kReleaseFuncIndex>(cur_run));
  const auto &executor = std::get<1>(cur_run);
  return std::make_pair(executor, nullptr);
}

void CustomV2AclnnKernelMod::InitInputType(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  static const std::map<std::string, CustomSupportType> support_types = {{"tensor", CustomSupportType::kTypeTensor},
                                                                         {"float", CustomSupportType::kTypeFloat},
                                                                         {"int", CustomSupportType::kTypeInt},
                                                                         {"bool", CustomSupportType::kTypeBool},
                                                                         {"string", CustomSupportType::kTypeString}};
  if (!primitive_->HasAttr(kCustomInputsType)) {
    MS_LOG(EXCEPTION) << "Please set attribute [inputs_type] for custom " << op_type_;
  }
  auto inputs_type = GetValue<std::string>(primitive_->GetAttr(kCustomInputsType));
  std::stringstream ss(inputs_type);
  std::string input_type;
  input_output_types_.clear();
  while (std::getline(ss, input_type, ',')) {
    MS_LOG(DEBUG) << "Input type: " << input_type;
    auto iter = support_types.find(input_type);
    if (iter == support_types.end()) {
      MS_LOG(EXCEPTION) << "Unsupported custom input type: " << input_type
                        << ", supported list: [tensor, float, int, bool, string]";
    }

    input_output_types_.emplace_back(iter->second);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    input_output_types_.emplace_back(CustomSupportType::kTypeTensor);
  }
}
}  // namespace custom
}  // namespace kernel
}  // namespace mindspore
