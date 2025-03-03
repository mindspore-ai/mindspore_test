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
#include "kernel/ascend/acl_ir/op_api_util.h"
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>
#include "acl/error_codes/rt_error_codes.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "mindspore/ops/op_def/math_op_name.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_compiler_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/device_context_conf/op_precision_conf.h"
#include "common/kernel_callback.h"

namespace mindspore::device::ascend {
namespace {
typedef aclError (*AclrtCtxSetSysParamOpt)(aclSysParamOpt, int64_t);

static const char k910BKey[] = "ascend910b";
static const char k310BKey[] = "ascend310b";
static const char k910_93Key[] = "ascend910_93";

static const std::unordered_map<std::string, aclCubeMathType> kCubeMathType = {
  {"force_fp16", FORCE_FP16},
  {"allow_fp32_to_fp16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision", ALLOW_FP32_DOWN_PRECISION},
  {"must_keep_origin_dtype", KEEP_DTYPE},
  {"allow_fp32_to_bf16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision_fp16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision_bf16", ALLOW_FP32_DOWN_PRECISION}};

static const std::unordered_map<uint8_t, aclCubeMathType> kSelectMoreMathType = {
  {0b01, KEEP_DTYPE}, {0b00, FORCE_FP16}, {0b11, FORCE_HF32}, {0b10, ALLOW_FP32_DOWN_PRECISION}};

static const std::unordered_map<std::string, bool> kMatmulEnableHf32 = {{"", false}, {"0", false}, {"1", true}};

static const std::unordered_map<std::string, bool> kConvEnableHf32 = {{"", true}, {"0", false}, {"1", true}};

std::mutex set_opt_mutex;

aclError SetCompileopt(aclCompileOpt opt, const char *value) { return CALL_ASCEND_API(aclSetCompileopt, opt, value); }

bool IsMatmulHf32Enable() {
  auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  auto allow_matmul_hf32 = op_precision_conf->matmul_allow_hf32();
  auto iter = kMatmulEnableHf32.find(allow_matmul_hf32);
  if (iter == kMatmulEnableHf32.end()) {
    MS_LOG(EXCEPTION) << "Unexpected config matmul_allow_hf32, which is " << allow_matmul_hf32;
  }
  return iter->second;
}

bool IsConvHf32Enable() {
  auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  auto allow_conv_hf32 = op_precision_conf->conv_allow_hf32();
  auto iter = kConvEnableHf32.find(allow_conv_hf32);
  if (iter == kConvEnableHf32.end()) {
    MS_LOG(EXCEPTION) << "Unexpected config conv_allow_hf32, which is " << allow_conv_hf32;
  }
  return iter->second;
}
}  // namespace

aclCubeMathType OpApiUtil::GetCubeMathType(bool use_hf32) {
  static std::string precision_mode = "not_inited";
  if (precision_mode == "not_inited") {
    auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
    MS_EXCEPTION_IF_NULL(op_precision_conf);
    precision_mode = op_precision_conf->precision_mode();
  }

  if (!precision_mode.empty() && kCubeMathType.count(precision_mode) != 0) {
    return kCubeMathType.at(precision_mode);
  }
  uint8_t select_mode = (static_cast<uint8_t>(use_hf32) << 1) + AclUtil::KeepOriginDType();
  if (kSelectMoreMathType.count(select_mode) != 0) {
    return kSelectMoreMathType.at(select_mode);
  }
  return AclUtil::KeepOriginDType() ? KEEP_DTYPE : ALLOW_FP32_DOWN_PRECISION;
}

bool OpApiUtil::IsAllowMatmulHF32() {
  static bool is_allow_matmul_hf32 = IsMatmulHf32Enable();
  return is_allow_matmul_hf32;
}

bool OpApiUtil::IsAllowConvHF32() {
  static bool is_allow_conv_hf32 = IsConvHf32Enable();
  return is_allow_conv_hf32;
}

void OpApiUtil::GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                        std::vector<std::string> *output_formats,
                                        std::vector<std::string> *input_reshape_types,
                                        std::vector<std::string> *output_reshape_types, const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);
  MS_EXCEPTION_IF_NULL(input_reshape_types);
  MS_EXCEPTION_IF_NULL(output_reshape_types);

  input_formats->clear();
  output_formats->clear();
  input_reshape_types->clear();
  output_reshape_types->clear();

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  size_t output_num = AnfUtils::GetOutputTensorNum(node);

  input_formats->assign(input_num, kOpFormat_DEFAULT);
  output_formats->assign(output_num, kOpFormat_DEFAULT);

  input_reshape_types->assign(input_num, "");
  output_reshape_types->assign(output_num, "");

  std::vector<size_t> special_inputs;
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    if (kernel_type == KernelType::GE_KERNEL) {
      (*input_formats)[i] = input_format;
    } else if (!AclHelper::CheckDefaultSupportFormat(input_format)) {
      (void)special_inputs.emplace_back(i);
    }
  }
  if (!special_inputs.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
  }
}

std::string OpApiUtil::GetCommName(const std::string &group) {
  static const auto get_comm_name =
    kernel::KernelCallback::GetInstance().GetCallback<std::string, const std::string &>("GetCommName");
  if (get_comm_name == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get GetCommNameCallback";
  }
  return get_comm_name(group);
}

bool OpApiUtil::NeedRebuildWorkspaceSize(const std::string &group, const std::string &inner_name) {
  if (!mindspore::UCEException::GetInstance().enable_arf()) {
    return false;
  }
  return OpApiUtil::GetCommName(group) != inner_name;
}

uint8_t AclUtil::KeepOriginDType() {
  static std::string version = "";
  static uint8_t need_keep_dtype = 0;
  if (version.empty()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    version = ms_context->ascend_soc_version();
    if (version.find(k910BKey) != std::string::npos || version.find(k310BKey) != std::string::npos ||
        version.find(k910_93Key) != std::string::npos) {
      need_keep_dtype = 1;
    }
  }
  return need_keep_dtype;
}

aclError AclUtil::SetCompileMode(const int64_t is_dynamic) {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  static int64_t last_mode = -1;
  if (is_dynamic != last_mode) {
    std::string mode = is_dynamic ? "disable" : "enable";
    auto set_compile_flag = SetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, mode.c_str());
    last_mode = is_dynamic;
    return set_compile_flag;
  }

  return ACL_SUCCESS;
}

aclError AclUtil::SetPrecisionMode(const std::string &mode) {
  std::lock_guard<std::mutex> lock(set_opt_mutex);

  static int8_t is_global_precision = -1;
  if (is_global_precision == -1) {
    auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
    MS_EXCEPTION_IF_NULL(op_precision_conf);
    auto precision_mode = op_precision_conf->precision_mode();
    if (!precision_mode.empty()) {
      is_global_precision = 1;
    } else {
      is_global_precision = 0;
    }
  }
  if (is_global_precision == 1) {
    return ACL_SUCCESS;
  }

  static std::string last_mode = (AclUtil::KeepOriginDType() == 1) ? "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  if (last_mode != mode) {
    auto ret = SetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, mode.c_str());
    last_mode = mode;
    return ret;
  }
  return ACL_SUCCESS;
}

void AclUtil::SetOpPrecisionMode() {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  auto op_precision_mode = op_precision_conf->op_precision_mode();
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(DEBUG) << "Set ACL_OP_PRECISION_MODE: " << op_precision_mode;
  auto ret = SetCompileopt(aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! error flag is " << ret;
  }
}
}  // namespace  mindspore::device::ascend
