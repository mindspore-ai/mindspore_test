/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/mkldnn/conv_grad_input_cpu_kernel.h"
#include <string>
#include <map>
#include <algorithm>
#include "mindspore/ops/op_def/conv_pool_op_name.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kConv2DBackpropInput = "Conv2DBackpropInput";
constexpr auto kConv3DBackpropInput = "Conv3DBackpropInput";
constexpr size_t kConvGradInputInputsMinNum = 2;
constexpr size_t kConvGradInputOutputsNum = 1;
}  // namespace

bool ConvGradInputCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ == kConv2DBackpropInputOpName) {
    weight_index_ = 1;
    diff_dst_index_ = 0;
  }
  group_ = GetValue<int64_t>(KernelMod::primitive_->GetAttr(GROUP));
  format_ = GetValue<std::string>(KernelMod::primitive_->GetAttr(FORMAT));
  if (format_ != NCHW && format_ != NCDHW) {
    MS_LOG(ERROR) << kernel_name_ << " only supports " << NCHW << " or " << NCDHW << " format "
                  << ", but got format: " << format_;
    return false;
  }
  const auto stride_attr = format_ == NCHW ? STRIDE : STRIDES;
  const auto dilation_attr = format_ == NCHW ? DILATION : DILATIONS;
  auto pad_mode_str = GetValue<std::string>(KernelMod::primitive_->GetAttr(PAD_MODE));
  std::map<std::string, mindspore::PadMode> str2padmode_map = {
    {PAD_MODE_LOWER_SAME, PadMode::SAME},   {PAD_MODE_UPPER_SAME, PadMode::SAME},
    {PAD_MODE_LOWER_VALID, PadMode::VALID}, {PAD_MODE_UPPER_VALID, PadMode::VALID},
    {PAD_MODE_LOWER_PAD, PadMode::PAD},     {PAD_MODE_UPPER_PAD, PadMode::PAD}};
  auto iter = str2padmode_map.find(pad_mode_str);
  if (iter == str2padmode_map.end()) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", pad_mode is illegal, got " << pad_mode_str;
  } else {
    pad_mode_ = iter->second;
  }
  strides_include_nc_ = GetValue<std::vector<int64_t>>(KernelMod::primitive_->GetAttr(stride_attr));
  dilation_include_nc_ = GetValue<std::vector<int64_t>>(KernelMod::primitive_->GetAttr(dilation_attr));
  return true;
}

int ConvGradInputCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> src_shape = outputs[0]->GetDeviceShapeVector();
  std::vector<int64_t> weight_shape = inputs[weight_index_]->GetDeviceShapeVector();
  std::vector<int64_t> dst_shape = inputs[diff_dst_index_]->GetDeviceShapeVector();
  size_t src_dim = src_shape.size();
  if (src_dim != weight_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input must be euqal to weight's, but got input shape: " << src_shape
                      << ", weight shape: " << weight_shape;
  }
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "Conv grad only supports 4D/5D input, but got " << src_dim << "D!";
  }
  const auto &format = format_;
  if (src_dim == SHAPE_4D && format != NCHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 4D input with NCHW format, but got format" << format;
  }
  if (src_dim == SHAPE_5D && format != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 5D input with NCDHW format, but got format " << format;
  }
  dnnl::memory::dims kernel_size(weight_shape.begin() + NC_LEN, weight_shape.end());
  const auto group = group_;
  if (group > 1) {
    if (src_shape[1] % group != 0) {
      MS_LOG(EXCEPTION) << "Conv grad channels must be divided by group!";
    }
    (void)weight_shape.insert(weight_shape.begin(), group);
    weight_shape[1] = weight_shape[1] / group;
  }
  const dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  const dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  const dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  const auto &strides_include_nc = strides_include_nc_;
  if (strides_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << "requires strides must be " << src_dim << "D, but got "
                      << strides_include_nc.size() << "D!";
  }
  const auto &dilation_include_nc = dilation_include_nc_;
  if (dilation_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires dilation must be " << src_dim << "D, but got "
                      << dilation_include_nc.size() << "D!";
  }
  const dnnl::memory::dims strides(strides_include_nc.begin() + NC_LEN, strides_include_nc.end());
  const dnnl::memory::dims dilation(dilation_include_nc.begin() + NC_LEN, dilation_include_nc.end());
  dnnl::memory::dims dilates;
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  (void)std::transform(dilation.begin(), dilation.end(), std::back_inserter(dilates),
                       [](const int64_t &value) { return value - 1; });
  const auto &pad_mode = pad_mode_;
  PaddingInfo padding_info{pad_mode, kernel_size, strides, dilation, &padding_l, &padding_r};
  GetPadding(src_shape, padding_info);

  const auto forward_desc = CreateDesc<dnnl::convolution_forward::desc>(
    dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides,
    dilates, padding_l, padding_r);
  const auto forward_prim_desc = CreateDesc<dnnl::convolution_forward::primitive_desc>(forward_desc, engine_);
  const auto backward_desc = CreateDesc<dnnl::convolution_backward_data::desc>(
    dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides, dilates, padding_l, padding_r);
  const auto backward_prim_desc =
    CreateDesc<dnnl::convolution_backward_data::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::convolution_backward_data>(backward_prim_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, dst_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  return KRET_OK;
}

bool ConvGradInputCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  if (inputs.size() < kConvGradInputInputsMinNum) {
    MS_LOG(EXCEPTION) << "Input numbers can not less " << kConvGradInputInputsMinNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConvGradInputOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[diff_dst_index_]->device_ptr());
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[weight_index_]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->device_ptr());
  ExecutePrimitive();
  return true;
}

std::vector<KernelAttr> ConvGradInputCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kConv2DBackpropInput,
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32)}},
    {kConv3DBackpropInput,
     {KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32)}}};
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "ConvGradInput does not support " << kernel_type_;
  }
  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conv2DBackpropInput,
                                 []() { return std::make_shared<ConvGradInputCpuKernelMod>(kConv2DBackpropInput); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conv3DBackpropInput,
                                 []() { return std::make_shared<ConvGradInputCpuKernelMod>(kConv3DBackpropInput); });
}  // namespace kernel
}  // namespace mindspore
