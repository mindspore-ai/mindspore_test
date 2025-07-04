/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "infer/grad/de_conv2d_grad_filter.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(DeConv2DGradFilter, BaseOperator);
void DeConv2DGradFilter::Init(const int64_t in_channel, const int64_t out_channel,
                              const std::vector<int64_t> &kernel_size, const PadMode &pad_mode,
                              const std::vector<int64_t> &pad_list, const std::vector<int64_t> &stride,
                              const std::vector<int64_t> &dilation, const int64_t group, const Format &format,
                              const ActivationType &activation_type, const bool has_bias) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_activation_type(activation_type);
  set_has_bias(has_bias);
}

void DeConv2DGradFilter::set_in_channel(const int64_t in_channel) {
  (void)this->AddAttr(kInChannel, api::MakeValue(in_channel));
}

int64_t DeConv2DGradFilter::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_out_channel(const int64_t out_channel) {
  (void)this->AddAttr(kOutChannel, api::MakeValue(out_channel));
}

int64_t DeConv2DGradFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> DeConv2DGradFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode DeConv2DGradFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> DeConv2DGradFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> DeConv2DGradFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, api::MakeValue(dilation));
}

std::vector<int64_t> DeConv2DGradFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_group(const int64_t group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

int64_t DeConv2DGradFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format DeConv2DGradFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

ActivationType DeConv2DGradFilter::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_has_bias(const bool has_bias) { (void)this->AddAttr(kHasBias, api::MakeValue(has_bias)); }

bool DeConv2DGradFilter::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameDeConv2DGradFilter, DeConv2DGradFilter);
}  // namespace ops
}  // namespace mindspore
