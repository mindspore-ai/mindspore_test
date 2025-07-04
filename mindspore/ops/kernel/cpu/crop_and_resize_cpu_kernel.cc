/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/crop_and_resize_cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "mindspore/ops/infer/crop_and_resize.h"

namespace mindspore {
namespace kernel {
namespace crop_and_resize_cpu {
bool CropAndResizeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);

  // suppose use kernel_ptr->get_method(), but the definition in lite is enumeration, not std::string. So we use this
  // for the moment to support dynamic shape.
  std::string method = GetValue<std::string>(primitive_->GetAttr("method"));
  if (method == "bilinear") {
    method_ = BILINEAR;
  } else if (method == "nearest") {
    method_ = NEAREST;
  } else {  //  bilinear-v2
    method_ = BILINEAR_V2;
  }
  extrapolation_value_ = GetValue<float>(primitive_->GetAttr(ops::kExtrapolationValue));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CropAndResizeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  //  input image
  auto input_image_shape = inputs[IMAGE]->GetShapeVector();
  size_t input_image_shape_len = input_image_shape.size();
  if (input_image_shape_len != IMAGE_DIM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'image' must be " << IMAGE_DIM << "-D, but got "
                  << input_image_shape_len << "-D.";
  }

  input_batch_ = LongToInt(input_image_shape[IMAGE_BATCH]);
  input_height_ = LongToInt(input_image_shape[IMAGE_HEIGHT]);
  input_width_ = LongToInt(input_image_shape[IMAGE_WEIGHT]);

  //  input boxes
  auto input_boxes_shape = inputs[BOXES]->GetShapeVector();
  size_t input_boxes_shape_len = input_boxes_shape.size();
  if (input_boxes_shape_len != BOX_RANK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'boxes' must be " << BOX_RANK << ", but got "
                  << input_boxes_shape_len;
  }

  //  input box_index
  auto input_box_index_shape = inputs[BOX_INDEX]->GetShapeVector();
  size_t input_box_index_shape_len = input_box_index_shape.size();
  if (input_box_index_shape_len != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'box_index' must be 1, but got "
                  << input_box_index_shape_len << ".";
  }

  //  input crop_size
  auto input_crop_size_shape = inputs[CROP_SIZE]->GetShapeVector();
  size_t input_crop_size_shape_len = input_crop_size_shape.size();
  if (input_crop_size_shape_len != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'crop_size' must be 1, but got "
                  << input_crop_size_shape_len << ".";
  }
  if (input_crop_size_shape[0] != CROP_SIZE_LEN) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the first dimension value of 'crop_size' must be " << CROP_SIZE_LEN
                  << ", but got " << input_crop_size_shape[0];
  }

  //  output
  constexpr size_t HEIGHT = 1;
  constexpr size_t WEIGHT = 2;
  constexpr size_t CHANNEL = 3;
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto output_shape_len = output_shape.size();
  output_size_ = 1;
  for (size_t i = 0; i < output_shape_len; i++) {
    output_size_ *= LongToInt(output_shape[i]);
  }

  //  set expected output params
  final_height_ = LongToInt(output_shape[HEIGHT]);
  final_width_ = LongToInt(output_shape[WEIGHT]);
  channel_ = LongToInt(output_shape[CHANNEL]);

  return KRET_OK;
}

template <typename T>
void CropAndResizeCpuKernelMod::BilinearResize(T *input_image, float target_x, float target_y, size_t pos,
                                               int box_index, int pos_channel, float *output) const {
  const int top_y_index = FloatToInt(floorf(target_y));
  const int bottom_y_index = FloatToInt(ceilf(target_y));
  const int left_x_index = FloatToInt(floorf(target_x));
  const int right_x_index = FloatToInt(ceilf(target_x));

  const float top_left = static_cast<float>(
    input_image[((box_index * input_height_ + top_y_index) * input_width_ + left_x_index) * channel_ + pos_channel]);
  const float top_right = static_cast<float>(
    input_image[((box_index * input_height_ + top_y_index) * input_width_ + right_x_index) * channel_ + pos_channel]);
  const float bottom_left = static_cast<float>(
    input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + left_x_index) * channel_ + pos_channel]);
  const float bottom_right = static_cast<float>(
    input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + right_x_index) * channel_ +
                pos_channel]);
  const float top = top_left + (top_right - top_left) * (target_x - left_x_index);
  const float bottom = bottom_left + (bottom_right - bottom_left) * (target_x - left_x_index);
  output[pos] = top + (bottom - top) * (target_y - top_y_index);
}

template <typename T>
void CropAndResizeCpuKernelMod::BilinearV2Resize(T *input_image, float y1, float x1, float y2, float x2, int pos_y,
                                                 int pos_x, size_t pos, int box_index, int pos_channel,
                                                 float *output) const {
  const float HALF = 0.5;
  int y1h = FloatToInt(y1 * input_height_);
  int x1w = FloatToInt(x1 * input_width_);
  int y2h = FloatToInt(y2 * input_height_);
  int x2w = FloatToInt(x2 * input_width_);
  int w = ((x2w - x1w + 1) > 1) ? x2w - x1w + 1 : 1;
  int h = ((y2h - y1h + 1) > 1) ? y2h - y1h + 1 : 1;

  float y_point = (pos_y + HALF) * (h / IntToFloat(final_height_)) - HALF;
  int top_y_index = std::min(std::max(0, FloatToInt(floorf(y_point))), h - 1);
  int bottom_y_index = std::min(std::max(0, FloatToInt(ceilf(y_point))), h - 1);

  float x_point = (pos_x + HALF) * (w / IntToFloat(final_width_)) - HALF;
  int left_x_index = std::min(std::max(0, FloatToInt(floorf(x_point))), w - 1);
  int right_x_index = std::min(std::max(0, FloatToInt(ceilf(x_point))), w - 1);

  const float y_lerp = y_point - top_y_index;
  const float x_lerp = x_point - left_x_index;

  const int y_top_index = std::max(0, y1h + top_y_index);
  const int y_bottom_index = std::max(0, y1h + bottom_y_index);
  const int x_left_index = std::max(0, x1w + left_x_index);
  const int x_right_index = std::max(0, x1w + right_x_index);

  const float top_left = static_cast<float>(
    input_image[((box_index * input_height_ + y_top_index) * input_width_ + x_left_index) * channel_ + pos_channel]);
  const float top_right = static_cast<float>(
    input_image[((box_index * input_height_ + y_top_index) * input_width_ + x_right_index) * channel_ + pos_channel]);
  const float bottom_left = static_cast<float>(
    input_image[((box_index * input_height_ + y_bottom_index) * input_width_ + x_left_index) * channel_ + pos_channel]);
  const float bottom_right = static_cast<float>(
    input_image[((box_index * input_height_ + y_bottom_index) * input_width_ + x_right_index) * channel_ +
                pos_channel]);

  output[pos] = top_left * (1 - y_lerp) * (1 - x_lerp) + bottom_right * y_lerp * x_lerp +
                top_right * (1 - y_lerp) * x_lerp + bottom_left * y_lerp * (1 - x_lerp);
}

template <typename T>
bool CropAndResizeCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_image = GetDeviceAddress<T>(inputs, IMAGE);
  auto *input_boxes = GetDeviceAddress<float>(inputs, BOXES);
  auto *input_box_index = GetDeviceAddress<int>(inputs, BOX_INDEX);
  auto *output = GetDeviceAddress<float>(outputs, kIndex0);

  auto task = [this, &input_box_index, &input_boxes, &input_image, &output](size_t start, size_t end) {
    const float HALF = 0.5;
    for (size_t pos = start; pos < end; pos++) {
      int pos_temp = SizeToInt(pos);
      const int pos_channel = pos_temp % channel_;
      pos_temp = pos_temp / channel_;
      const int pos_x = pos_temp % final_width_;
      pos_temp = pos_temp / final_width_;
      const int pos_y = pos_temp % final_height_;
      const int pos_image_idx = pos_temp / final_height_;
      const int box_index = input_box_index[pos_image_idx];
      if (box_index < 0 || box_index >= input_batch_) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the value of box_index must be in [0, "
                                 << input_batch_ << "), but got [" << box_index << "].";
      }

      //  crop values
      const float y1 = input_boxes[4 * pos_image_idx];
      const float x1 = input_boxes[4 * pos_image_idx + 1];
      const float y2 = input_boxes[4 * pos_image_idx + 2];
      const float x2 = input_boxes[4 * pos_image_idx + 3];

      //  set scale and target pixels
      float scale_height = final_height_ > 1 ? (y2 - y1) * (input_height_ - 1) / (final_height_ - 1) : 0;
      float scale_width = final_width_ > 1 ? (x2 - x1) * (input_width_ - 1) / (final_width_ - 1) : 0;
      float target_y =
        final_height_ > 1 ? y1 * (input_height_ - 1) + pos_y * scale_height : HALF * (y1 + y2) * (input_height_ - 1);
      float target_x =
        final_width_ > 1 ? x1 * (input_width_ - 1) + pos_x * scale_width : HALF * (x1 + x2) * (input_width_ - 1);

      //  use extrapolation value if out of range
      if (((target_x < 0) || (target_x > input_width_ - 1)) || ((target_y < 0) || (target_y > input_height_ - 1))) {
        if ((method_ == BILINEAR) || (method_ == NEAREST)) {
          output[pos] = extrapolation_value_;
          continue;
        }
      }

      if (method_ == BILINEAR) {
        // Bilinear
        BilinearResize<T>(input_image, target_x, target_y, pos, box_index, pos_channel, output);
      } else if (method_ == BILINEAR_V2) {
        BilinearV2Resize(input_image, y1, x1, y2, x2, pos_y, pos_x, pos, box_index, pos_channel, output);
        // BilinearV2
      } else {
        // Nearest Neighbour
        const int closest_x_index = FloatToInt(roundf(target_x));
        const int closest_y_index = FloatToInt(roundf(target_y));
        const float val = static_cast<float>(
          input_image[((box_index * input_height_ + closest_y_index) * input_width_ + closest_x_index) * channel_ +
                      pos_channel]);
        output[pos] = val;
      }
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(output_size_), this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, CropAndResizeCpuKernelMod::CropAndResizeFunc>> CropAndResizeCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<uint16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &CropAndResizeCpuKernelMod::LaunchKernel<uint16_t>}};

std::vector<KernelAttr> CropAndResizeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CropAndResizeFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CropAndResize, CropAndResizeCpuKernelMod);
}  // namespace crop_and_resize_cpu
}  // namespace kernel
}  // namespace mindspore
