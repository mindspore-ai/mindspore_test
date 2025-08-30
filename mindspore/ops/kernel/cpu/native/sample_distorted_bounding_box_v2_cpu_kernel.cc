/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/sample_distorted_bounding_box_v2_cpu_kernel.h"
#include <random>
#include "mindspore/ops/infer/sample_distorted_bounding_box_v2.h"

#include "kernel/cpu/mkldnn/mkl_cpu_kernel.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace sample_distorted_bounding_box_v2_cpu {
namespace {
constexpr size_t kOutputSize = 3;
constexpr size_t kInputSize = 3;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kBBoxesDimension = 3;
constexpr size_t kShapeSize1 = 1;
constexpr size_t kShapeSize2 = 2;
constexpr size_t kShapeSize3 = 3;
constexpr size_t kShapeSize4 = 4;
constexpr size_t kNumber0 = 0;
constexpr float kFloatNum0 = 0.0;
constexpr float kFloatNum1 = 1.0;

inline bool IsValidRandomCropParams(int original_width, int original_height, float min_relative_crop_area,
                                    float max_relative_crop_area, float aspect_ratio) {
  if (original_width <= 0 || original_height <= 0) {
    return false;
  }
  if (max_relative_crop_area <= 0.0f) {
    return false;
  }
  if (aspect_ratio <= 0.0f) {
    return false;
  }
  if (min_relative_crop_area > max_relative_crop_area) {
    return false;
  }
  return true;
}

inline int AdjustMaxHeightByWidth(int max_height, float aspect_ratio, int original_width) {
  const float kEps = 0.0000001f;
  const float kRoundBias = 0.5f;
  if (lrintf(static_cast<float>(max_height) * aspect_ratio) > original_width) {
    int adjusted = static_cast<int>((original_width + kRoundBias - kEps) / aspect_ratio);
    if (lrintf(static_cast<float>(adjusted) * aspect_ratio) > original_width) {
      adjusted -= 1;
    }
    return adjusted;
  }
  return max_height;
}

inline void AdjustHeightWidthToArea(int *height, int *width, float *area, float aspect_ratio, float min_area,
                                    float max_area) {
  *width = static_cast<int>(lrintf(static_cast<float>(*height) * aspect_ratio));
  *area = static_cast<float>(*width) * static_cast<float>(*height);
  if (*area < min_area) {
    ++(*height);
    *width = static_cast<int>(lrintf(static_cast<float>(*height) * aspect_ratio));
    *area = static_cast<float>(*width) * static_cast<float>(*height);
  } else if (*area > max_area) {
    --(*height);
    *width = static_cast<int>(lrintf(static_cast<float>(*height) * aspect_ratio));
    *area = static_cast<float>(*width) * static_cast<float>(*height);
  }
}

inline bool IsGeometryValid(int width, int height, float area, int original_width, int original_height, float min_area,
                            float max_area) {
  if (area < min_area || area > max_area) {
    return false;
  }
  if (width > original_width || height > original_height) {
    return false;
  }
  if (width <= 0 || height <= 0) {
    return false;
  }
  return true;
}
}  // namespace

const uint64_t SampleDistortedBoundingBoxV2CPUKernelMod::New64() {
  std::random_device device("/dev/urandom");
  static std::mt19937_64 rng = std::mt19937_64(device());
  return (rng)();
}

void SampleDistortedBoundingBoxV2CPUKernelMod::InitMSPhiloxRandom(int64_t seed, int64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = static_cast<int64_t>(New64());
    seed2 = static_cast<int64_t>(New64());
  }
  generator_ = random::PhiloxRandom(seed, seed2);
}

float SampleDistortedBoundingBoxV2CPUKernelMod::RandFloat() {
  uint32_t x = GenerateSingle();
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  float result;
  int ret = memcpy_s(&result, sizeof(result), &val, sizeof(val));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  return result - 1.0f;
}

uint32_t SampleDistortedBoundingBoxV2CPUKernelMod::Uniform(uint32_t n) {
  if (n == 0) {
    return 0;
  } else if (0 == (n & (n - 1))) {
    return GenerateSingle() & (n - 1);
  } else {
    const uint32_t range = ~static_cast<uint32_t>(0);
    const uint32_t rem = (range % n) + 1;
    uint32_t rnd;
    do {
      rnd = GenerateSingle();
    } while (rnd < rem);
    return rnd % n;
  }
}

uint32_t SampleDistortedBoundingBoxV2CPUKernelMod::GenerateSingle() {
  if (used_result_index_ == random::PhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

bool SampleDistortedBoundingBoxV2CPUKernelMod::SatisfiesOverlapConstraints(
  const Region &crop, float minimum_object_covered, const std::vector<Region> &bounding_boxes) const {
  constexpr float kMinValidArea = 1.0f;
  const float crop_area = crop.Area();
  if (crop_area < kMinValidArea) {
    return false;
  }

  // Check whether the crop covers at least one object by the required fraction.
  for (const auto &bbox : bounding_boxes) {
    const float object_area = bbox.Area();
    if (object_area < kMinValidArea) {
      continue;
    }
    const float intersection_area = crop.Intersect(bbox).Area();
    if (intersection_area >= minimum_object_covered * object_area) {
      return true;
    }
  }
  return false;
}

void SampleDistortedBoundingBoxV2CPUKernelMod::PickRandomOffsets(int width, int height, int original_width,
                                                                 int original_height, int *x, int *y) {
  *x = width < original_width ? static_cast<int>(Uniform(static_cast<uint32_t>(original_width - width))) : 0;
  *y = height < original_height ? static_cast<int>(Uniform(static_cast<uint32_t>(original_height - height))) : 0;
}

bool SampleDistortedBoundingBoxV2CPUKernelMod::GenerateRandomCrop(int ms_original_width, int ms_original_height,
                                                                  float ms_min_relative_crop_area,
                                                                  float ms_max_relative_crop_area,
                                                                  float ms_aspect_ratio, Region *ms_crop_rect) {
  if (!IsValidRandomCropParams(ms_original_width, ms_original_height, ms_min_relative_crop_area,
                               ms_max_relative_crop_area, ms_aspect_ratio)) {
    return false;
  }

  const float ms_min_area = ms_min_relative_crop_area * ms_original_width * ms_original_height;
  const float ms_max_area = ms_max_relative_crop_area * ms_original_width * ms_original_height;

  int height = static_cast<int>(lrintf(std::sqrt(ms_min_area / ms_aspect_ratio)));
  int max_height = static_cast<int>(lrintf(std::sqrt(ms_max_area / ms_aspect_ratio)));
  max_height = AdjustMaxHeightByWidth(max_height, ms_aspect_ratio, ms_original_width);

  max_height = std::min(max_height, ms_original_height);
  height = std::min(height, max_height);
  if (height < max_height) {
    height += static_cast<int>(Uniform(static_cast<uint32_t>(max_height - height + 1)));
  }

  int width = 0;
  float area = 0.0f;
  AdjustHeightWidthToArea(&height, &width, &area, ms_aspect_ratio, ms_min_area, ms_max_area);

  if (!IsGeometryValid(width, height, area, ms_original_width, ms_original_height, ms_min_area, ms_max_area)) {
    return false;
  }

  int x = 0;
  int y = 0;
  PickRandomOffsets(width, height, ms_original_width, ms_original_height, &x, &y);

  ms_crop_rect->x_min_ = x;
  ms_crop_rect->y_min_ = y;
  ms_crop_rect->x_max_ = x + width;
  ms_crop_rect->y_max_ = y + height;
  return true;
}

bool SampleDistortedBoundingBoxV2CPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  seed_ = GetValue<int64_t>(primitive_->GetAttr("seed"));
  seed2_ = GetValue<int64_t>(primitive_->GetAttr("seed2"));
  aspect_ratio_range_ = GetValue<std::vector<float>>(primitive_->GetAttr("aspect_ratio_range"));
  if (aspect_ratio_range_.size() != kShapeSize2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', aspect_ratio_range field must specify 2 dimensions.";
    return false;
  }
  if (aspect_ratio_range_[kIndex1] <= kFloatNum0 || aspect_ratio_range_[kIndex0] <= kFloatNum0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', aspect_ratio_range must be positive: ["
                  << aspect_ratio_range_[kIndex0] << "], [" << aspect_ratio_range_[kIndex1] << "].";
    return false;
  }
  area_range_ = GetValue<std::vector<float>>(primitive_->GetAttr("area_range"));
  if (area_range_.size() != kShapeSize2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', area_range field must specify 2 dimensions.";
    return false;
  }
  if (area_range_[kIndex1] <= kFloatNum0 || area_range_[kIndex0] <= kFloatNum0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', area_range must be positive: [" << area_range_[kIndex0] << "], ["
                  << area_range_[kIndex1] << "].";
    return false;
  }
  if (area_range_[kIndex1] > kFloatNum1 || area_range_[kIndex0] > kFloatNum1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', area_range must be less then or equal to 1.0: ["
                  << area_range_[kIndex0] << "], [" << area_range_[kIndex1] << "].";
    return false;
  }
  max_attempts_ = GetValue<int64_t>(primitive_->GetAttr("max_attempts"));
  if (max_attempts_ <= SizeToLong(kNumber0)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', max_attempts must be positive: [" << max_attempts_ << "].";
    return false;
  }
  use_image_if_no_bounding_boxes_ = GetValue<bool>(primitive_->GetAttr("use_image_if_no_bounding_boxes"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For 'Arithmetic', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  return true;
}

int SampleDistortedBoundingBoxV2CPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kIndex0]->dtype_id();
  auto shape_image_size = inputs[kIndex0]->GetDeviceShapeVector();
  auto shape_bounding_boxes = inputs[kIndex1]->GetDeviceShapeVector();
  size_t shape_dim_image_size = shape_image_size.size();
  size_t shape_dim_bounding_boxes = shape_bounding_boxes.size();

  if (shape_dim_image_size != kShapeSize1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', image_size must be 1-dimensional, got: [" << shape_dim_image_size
                  << "].";
    return KRET_RESIZE_FAILED;
  }
  if (LongToSize(shape_image_size[kIndex0]) != kShapeSize3) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', image_size must contain 3 elements, got: ["
                  << shape_image_size[kIndex0] << "].";
    return KRET_RESIZE_FAILED;
  }
  if (shape_dim_bounding_boxes != kBBoxesDimension) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', bounding_boxes must be 3-dimensional"
                  << " [batch, num_boxes, coords], got: [" << shape_dim_bounding_boxes << "].";
    return KRET_RESIZE_FAILED;
  }
  if (LongToSize(shape_bounding_boxes[shape_dim_bounding_boxes - 1]) != kShapeSize4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', bounding_boxes must have shape [4], got: ["
                  << shape_bounding_boxes[shape_dim_bounding_boxes - 1] << "].";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool SampleDistortedBoundingBoxV2CPUKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                                      const std::vector<kernel::KernelTensor *> & /* workspace */,
                                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex2]);
  if (dtype_ == kNumberTypeUInt8) {
    LaunchSDBBExt2<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchSDBBExt2<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchSDBBExt2<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchSDBBExt2<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchSDBBExt2<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', kernel data type " << TypeIdLabel(dtype_) << " not support.";
  }
  return true;
}

template <typename T>
void SampleDistortedBoundingBoxV2CPUKernelMod::LaunchSDBBExt2(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  auto image_size = GetDeviceAddress<T>(inputs, kIndex0);
  auto bounding_boxes = GetDeviceAddress<float>(inputs, kIndex1);
  auto min_object_covered = GetDeviceAddress<float>(inputs, kIndex2);
  auto begin = GetDeviceAddress<T>(outputs, kIndex0);
  auto size = GetDeviceAddress<T>(outputs, kIndex1);
  auto bboxes = GetDeviceAddress<float>(outputs, kIndex2);

  const int32_t height = static_cast<int32_t>(image_size[kIndex0]);
  const int32_t width = static_cast<int32_t>(image_size[kIndex1]);
  if (!(height > 0 && width > 0)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', image height and width must be positive, got: [" << height
                      << "] and [" << width << "].";
  }

  float min_object_covered_val = 0.0;
  min_object_covered_val = *min_object_covered;
  if (min_object_covered_val < 0.0 || min_object_covered_val > 1.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', min_object_covered must be in [0.0, 1.0], got: ["
                      << min_object_covered_val << "].";
  }

  std::vector<Region> boxes;
  size_t size_bounding_boxes = inputs[kIndex1]->size() / sizeof(float);
  for (size_t b = 0; b < size_bounding_boxes / kShapeSize4; ++b) {
    for (size_t i = 0; i < kShapeSize4; ++i) {
      if (bounding_boxes[b * kShapeSize4 + i] < 0.0 || bounding_boxes[b * kShapeSize4 + i] > 1.0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', all bounding box coordinates must in [0.0, 1.0], got: ["
                          << bounding_boxes[b * kShapeSize4 + i] << "].";
      }
    }
    if (!(bounding_boxes[b * kShapeSize4 + kIndex1] < bounding_boxes[b * kShapeSize4 + kIndex3])) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', x_min of bounding box must be less than x_max, got: ["
                        << bounding_boxes[b * kShapeSize4 + kIndex1] << "] and ["
                        << bounding_boxes[b * kShapeSize4 + kIndex3] << "].";
    }
    if (!(bounding_boxes[b * kShapeSize4 + kIndex0] < bounding_boxes[b * kShapeSize4 + kIndex2])) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', y_min of bounding box must be less than y_max, got: ["
                        << bounding_boxes[b * kShapeSize4 + kIndex0] << "] and ["
                        << bounding_boxes[b * kShapeSize4 + kIndex2] << "].";
    }
    const int32_t x_min = static_cast<int32_t>(bounding_boxes[b * kShapeSize4 + 1] * width);
    const int32_t y_min = static_cast<int32_t>(bounding_boxes[b * kShapeSize4 + 0] * height);
    const int32_t x_max = static_cast<int32_t>(bounding_boxes[b * kShapeSize4 + 3] * width);
    const int32_t y_max = static_cast<int32_t>(bounding_boxes[b * kShapeSize4 + 2] * height);
    (void)boxes.emplace_back(Region(x_min, y_min, x_max, y_max));
  }

  const Region ms_image_rect(0, 0, width, height);
  if (boxes.empty()) {
    if (!use_image_if_no_bounding_boxes_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', no bounding boxes provided as input. One must enable use_image_if_no_bounding_boxes "
                           "if you wish to not provide any bounding boxes.";
    }

    boxes.push_back(ms_image_rect);
  }

  const float ms_min_sample_area = area_range_[kIndex0];
  const float ms_max_sample_area = area_range_[kIndex1];
  const float ms_min_sample_aspect_ratio = aspect_ratio_range_[kIndex0];
  const float ms_max_sample_aspect_ratio = aspect_ratio_range_[kIndex1];

  InitMSPhiloxRandom(seed_, seed2_);

  Region ms_crop_rect;
  bool ms_sample_generated = false;
  for (size_t i = 0; i < LongToSize(max_attempts_); ++i) {
    const float sample_aspect_ratio =
      RandFloat() * (ms_max_sample_aspect_ratio - ms_min_sample_aspect_ratio) + ms_min_sample_aspect_ratio;
    if (GenerateRandomCrop(width, height, ms_min_sample_area, ms_max_sample_area, sample_aspect_ratio, &ms_crop_rect)) {
      if (SatisfiesOverlapConstraints(ms_crop_rect, min_object_covered_val, boxes)) {
        ms_sample_generated = true;
        break;
      }
    }
  }

  if (!ms_sample_generated) {
    ms_crop_rect = ms_image_rect;
  }

  // Determine the cropping parameters from the bounding box.
  const int target_width = ms_crop_rect.x_max_ - ms_crop_rect.x_min_;
  const int target_height = ms_crop_rect.y_max_ - ms_crop_rect.y_min_;
  const int offset_width = ms_crop_rect.x_min_;
  const int offset_height = ms_crop_rect.y_min_;

  if (width < target_width + offset_width) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', width must be >= target_width + offset_width: [" << width
                      << "] vs [" << target_width << "] + [" << offset_width << "]";
  }

  if (height < target_height + offset_height) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', height must be >= target_height + offset_height: [" << height
                      << "] vs [" << target_height << "] + [" << offset_height << "]";
  }

  begin[kIndex0] = static_cast<T>(offset_height);
  size[kIndex0] = static_cast<T>(target_height);
  begin[kIndex1] = static_cast<T>(offset_width);
  size[kIndex1] = static_cast<T>(target_width);

  bboxes[kIndex0] = static_cast<float>(ms_crop_rect.y_min_) / static_cast<float>(height);
  bboxes[kIndex1] = static_cast<float>(ms_crop_rect.x_min_) / static_cast<float>(width);
  bboxes[kIndex2] = static_cast<float>(ms_crop_rect.y_max_) / static_cast<float>(height);
  bboxes[kIndex3] = static_cast<float>(ms_crop_rect.x_max_) / static_cast<float>(width);

  // Retain all of the channels.
  begin[kIndex2] = static_cast<T>(0);
  size[kIndex2] = static_cast<T>(-1);
}

std::vector<KernelAttr> SampleDistortedBoundingBoxV2CPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeUInt8)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeUInt8)
                                                       .AddOutputAttr(kNumberTypeUInt8)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt8)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt8)
                                                       .AddOutputAttr(kNumberTypeInt8)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt16)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt16)
                                                       .AddOutputAttr(kNumberTypeInt16)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SampleDistortedBoundingBoxV2, SampleDistortedBoundingBoxV2CPUKernelMod);
}  // namespace sample_distorted_bounding_box_v2_cpu
}  // namespace kernel
}  // namespace mindspore
