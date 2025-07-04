/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/transforms.h"
#if defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#endif
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/pyav/container.h"
#include "minddata/dataset/kernels/image/pyav/context.h"
#include "minddata/dataset/kernels/image/pyav/frame.h"
#include "minddata/dataset/kernels/image/pyav/packet.h"
#include "minddata/dataset/kernels/image/pyav/stream.h"
#include "minddata/dataset/kernels/image/video_utils.h"

#include "minddata/dataset/kernels/ir/vision/adjust_brightness_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_gamma_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_hue_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_saturation_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/convert_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_video_ir.h"
#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/erase_ir.h"
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"
#include "minddata/dataset/kernels/ir/vision/horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_to_size_ir.h"
#include "minddata/dataset/kernels/ir/vision/perspective_ir.h"
#include "minddata/dataset/kernels/ir/vision/posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/rand_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_adjust_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_lighting_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"
#include "minddata/dataset/kernels/ir/vision/slice_patches_ir.h"
#include "minddata/dataset/kernels/ir/vision/solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/to_tensor_ir.h"
#include "minddata/dataset/kernels/ir/vision/trivial_augment_wide_ir.h"
#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"
#include "minddata/dataset/kernels/ir/vision/vertical_flip_ir.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(AdjustBrightnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::AdjustBrightnessOperation, TensorOperation,
                                   std::shared_ptr<vision::AdjustBrightnessOperation>>(*m, "AdjustBrightnessOperation")
                    .def(py::init([](float brightness_factor, const std::string &device_target) {
                      auto adjust_brightness =
                        std::make_shared<vision::AdjustBrightnessOperation>(brightness_factor, device_target);
                      THROW_IF_ERROR(adjust_brightness->ValidateParams());
                      return adjust_brightness;
                    }));
                }));

PYBIND_REGISTER(AdjustContrastOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::AdjustContrastOperation, TensorOperation,
                                   std::shared_ptr<vision::AdjustContrastOperation>>(*m, "AdjustContrastOperation")
                    .def(py::init([](float contrast_factor, const std::string &device_target) {
                      auto adjust_contrast =
                        std::make_shared<vision::AdjustContrastOperation>(contrast_factor, device_target);
                      THROW_IF_ERROR(adjust_contrast->ValidateParams());
                      return adjust_contrast;
                    }));
                }));

PYBIND_REGISTER(
  AdjustGammaOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AdjustGammaOperation, TensorOperation, std::shared_ptr<vision::AdjustGammaOperation>>(
      *m, "AdjustGammaOperation")
      .def(py::init([](float gamma, float gain) {
        auto ajust_gamma = std::make_shared<vision::AdjustGammaOperation>(gamma, gain);
        THROW_IF_ERROR(ajust_gamma->ValidateParams());
        return ajust_gamma;
      }));
  }));

PYBIND_REGISTER(
  AdjustHueOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AdjustHueOperation, TensorOperation, std::shared_ptr<vision::AdjustHueOperation>>(
      *m, "AdjustHueOperation")
      .def(py::init([](float hue_factor, const std::string &device_target) {
        auto adjust_hue = std::make_shared<vision::AdjustHueOperation>(hue_factor, device_target);
        THROW_IF_ERROR(adjust_hue->ValidateParams());
        return adjust_hue;
      }));
  }));

PYBIND_REGISTER(AdjustSaturationOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::AdjustSaturationOperation, TensorOperation,
                                   std::shared_ptr<vision::AdjustSaturationOperation>>(*m, "AdjustSaturationOperation")
                    .def(py::init([](float saturation_factor, const std::string &device_target) {
                      auto ajust_saturation =
                        std::make_shared<vision::AdjustSaturationOperation>(saturation_factor, device_target);
                      THROW_IF_ERROR(ajust_saturation->ValidateParams());
                      return ajust_saturation;
                    }));
                }));

PYBIND_REGISTER(AdjustSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::AdjustSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::AdjustSharpnessOperation>>(*m, "AdjustSharpnessOperation")
                    .def(py::init([](float sharpness_factor, const std::string &device_target) {
                      auto adjust_sharpness =
                        std::make_shared<vision::AdjustSharpnessOperation>(sharpness_factor, device_target);
                      THROW_IF_ERROR(adjust_sharpness->ValidateParams());
                      return adjust_sharpness;
                    }));
                }));

PYBIND_REGISTER(AffineOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::AffineOperation, TensorOperation, std::shared_ptr<vision::AffineOperation>>(
                    *m, "AffineOperation")
                    .def(py::init([](float degrees, const std::vector<float> &translation, float scale,
                                     const std::vector<float> &shear, InterpolationMode interpolation,
                                     const std::vector<uint8_t> &fill_value, const std::string &device_target) {
                      auto affine = std::make_shared<vision::AffineOperation>(degrees, translation, scale, shear,
                                                                              interpolation, fill_value, device_target);
                      THROW_IF_ERROR(affine->ValidateParams());
                      return affine;
                    }));
                }));

PYBIND_REGISTER(
  AutoAugmentOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AutoAugmentOperation, TensorOperation, std::shared_ptr<vision::AutoAugmentOperation>>(
      *m, "AutoAugmentOperation")
      .def(
        py::init([](AutoAugmentPolicy policy, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
          auto auto_augment = std::make_shared<vision::AutoAugmentOperation>(policy, interpolation, fill_value);
          THROW_IF_ERROR(auto_augment->ValidateParams());
          return auto_augment;
        }));
  }));

PYBIND_REGISTER(
  AutoContrastOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AutoContrastOperation, TensorOperation, std::shared_ptr<vision::AutoContrastOperation>>(
      *m, "AutoContrastOperation")
      .def(py::init([](float cutoff, const std::vector<uint32_t> &ignore, const std::string &device_target) {
        auto auto_contrast = std::make_shared<vision::AutoContrastOperation>(cutoff, ignore, device_target);
        THROW_IF_ERROR(auto_contrast->ValidateParams());
        return auto_contrast;
      }));
  }));

PYBIND_REGISTER(BoundingBoxAugmentOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::BoundingBoxAugmentOperation, TensorOperation,
                                   std::shared_ptr<vision::BoundingBoxAugmentOperation>>(*m,
                                                                                         "BoundingBoxAugmentOperation")
                    .def(py::init([](const py::object &transform, float ratio) {
                      auto bounding_box_augment = std::make_shared<vision::BoundingBoxAugmentOperation>(
                        std::move(toTensorOperation(transform)), ratio);
                      THROW_IF_ERROR(bounding_box_augment->ValidateParams());
                      return bounding_box_augment;
                    }));
                }));

PYBIND_REGISTER(
  CenterCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CenterCropOperation, TensorOperation, std::shared_ptr<vision::CenterCropOperation>>(
      *m, "CenterCropOperation", "Tensor operation to crop and image in the middle. Takes height and width (optional)")
      .def(py::init([](const std::vector<int32_t> &size) {
        auto center_crop = std::make_shared<vision::CenterCropOperation>(size);
        THROW_IF_ERROR(center_crop->ValidateParams());
        return center_crop;
      }));
  }));

PYBIND_REGISTER(
  ConvertColorOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::ConvertColorOperation, TensorOperation, std::shared_ptr<vision::ConvertColorOperation>>(
      *m, "ConvertColorOperation", "Tensor operation to change the color space of the image.")
      .def(py::init([](ConvertMode convert_mode, const std::string &device_target) {
        auto convert = std::make_shared<vision::ConvertColorOperation>(convert_mode, device_target);
        THROW_IF_ERROR(convert->ValidateParams());
        return convert;
      }));
  }));

PYBIND_REGISTER(CropOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CropOperation, TensorOperation, std::shared_ptr<vision::CropOperation>>(
                    *m, "CropOperation", "Tensor operation to crop images")
                    .def(py::init([](std::vector<int32_t> coordinates, const std::vector<int32_t> &size,
                                     const std::string &device_target) {
                      // In Python API, the order of coordinates is first top then left, which is different from
                      // those in CropOperation. So we need to swap the coordinates.
                      std::swap(coordinates[0], coordinates[1]);
                      auto crop = std::make_shared<vision::CropOperation>(coordinates, size, device_target);
                      THROW_IF_ERROR(crop->ValidateParams());
                      return crop;
                    }));
                }));

PYBIND_REGISTER(
  CutMixBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CutMixBatchOperation, TensorOperation, std::shared_ptr<vision::CutMixBatchOperation>>(
      *m, "CutMixBatchOperation", "Tensor operation to cutmix a batch of images")
      .def(py::init([](ImageBatchFormat image_batch_format, float alpha, float prob) {
        auto cut_mix_batch = std::make_shared<vision::CutMixBatchOperation>(image_batch_format, alpha, prob);
        THROW_IF_ERROR(cut_mix_batch->ValidateParams());
        return cut_mix_batch;
      }));
  }));

PYBIND_REGISTER(CutOutOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CutOutOperation, TensorOperation, std::shared_ptr<vision::CutOutOperation>>(
                    *m, "CutOutOperation",
                    "Tensor operation to randomly erase a portion of the image. Takes height and width.")
                    .def(py::init([](int32_t length, int32_t num_patches, bool is_hwc) {
                      auto cut_out = std::make_shared<vision::CutOutOperation>(length, num_patches, is_hwc);
                      THROW_IF_ERROR(cut_out->ValidateParams());
                      return cut_out;
                    }));
                }));

PYBIND_REGISTER(DecodeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::DecodeOperation, TensorOperation, std::shared_ptr<vision::DecodeOperation>>(
                    *m, "DecodeOperation")
                    .def(py::init([](bool rgb, const std::string &device_target) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb, device_target);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }));
                }));

#ifdef ENABLE_FFMPEG
PYBIND_REGISTER(
  DecodeVideoOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::DecodeVideoOperation, TensorOperation, std::shared_ptr<vision::DecodeVideoOperation>>(
      *m, "DecodeVideoOperation")
      .def(py::init([]() {
        auto decode_video = std::make_shared<vision::DecodeVideoOperation>();
        THROW_IF_ERROR(decode_video->ValidateParams());
        return decode_video;
      }));
  }));
#endif

PYBIND_REGISTER(EncodeJpegOperation, 1, ([](py::module *m) {
                  (void)m->def("encode_jpeg", ([](const std::shared_ptr<Tensor> &image, int quality) {
                                 std::shared_ptr<Tensor> output;
                                 THROW_IF_ERROR(EncodeJpeg(image, &output, quality));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(EncodePNGOperation, 1, ([](py::module *m) {
                  (void)m->def("encode_png", ([](const std::shared_ptr<Tensor> &image, int compression_level) {
                                 std::shared_ptr<Tensor> output;
                                 THROW_IF_ERROR(EncodePng(image, &output, compression_level));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(EqualizeOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::EqualizeOperation, TensorOperation, std::shared_ptr<vision::EqualizeOperation>>(
                      *m, "EqualizeOperation")
                      .def(py::init([](const std::string &device_target) {
                        auto equalize = std::make_shared<vision::EqualizeOperation>(device_target);
                        THROW_IF_ERROR(equalize->ValidateParams());
                        return equalize;
                      }));
                }));

PYBIND_REGISTER(EraseOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::EraseOperation, TensorOperation, std::shared_ptr<vision::EraseOperation>>(
                    *m, "EraseOperation")
                    .def(py::init([](int32_t top, int32_t left, int32_t height, int32_t width,
                                     const std::vector<float> &value, bool inplace, const std::string &device_target) {
                      auto erase = std::make_shared<vision::EraseOperation>(top, left, height, width, value, inplace,
                                                                            device_target);
                      THROW_IF_ERROR(erase->ValidateParams());
                      return erase;
                    }));
                }));

PYBIND_REGISTER(
  GaussianBlurOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::GaussianBlurOperation, TensorOperation, std::shared_ptr<vision::GaussianBlurOperation>>(
      *m, "GaussianBlurOperation")
      .def(py::init(
        [](const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma, const std::string &device_target) {
          auto gaussian_blur = std::make_shared<vision::GaussianBlurOperation>(kernel_size, sigma, device_target);
          THROW_IF_ERROR(gaussian_blur->ValidateParams());
          return gaussian_blur;
        }));
  }));

PYBIND_REGISTER(GetImageNumChannels, 1, ([](py::module *m) {
                  (void)m->def("get_image_num_channels", ([](const std::shared_ptr<Tensor> &image) {
                                 dsize_t channels;
                                 THROW_IF_ERROR(ImageNumChannels(image, &channels));
                                 return channels;
                               }));
                }));

PYBIND_REGISTER(GetImageSize, 1, ([](py::module *m) {
                  (void)m->def("get_image_size", ([](const std::shared_ptr<Tensor> &image) {
                                 auto size = std::vector<dsize_t>(2);
                                 THROW_IF_ERROR(ImageSize(image, &size));
                                 py::list size_list;
                                 size_list.append(size[0]);
                                 size_list.append(size[1]);
                                 return size_list;
                               }));
                }));

PYBIND_REGISTER(HorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::HorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::HorizontalFlipOperation>>(*m, "HorizontalFlipOperation")
                    .def(py::init([](const std::string &device_target) {
                      auto horizontal_flip = std::make_shared<vision::HorizontalFlipOperation>(device_target);
                      THROW_IF_ERROR(horizontal_flip->ValidateParams());
                      return horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(HwcToChwOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::HwcToChwOperation, TensorOperation, std::shared_ptr<vision::HwcToChwOperation>>(
                      *m, "HwcToChwOperation")
                      .def(py::init([]() {
                        auto hwc_to_chw = std::make_shared<vision::HwcToChwOperation>();
                        THROW_IF_ERROR(hwc_to_chw->ValidateParams());
                        return hwc_to_chw;
                      }));
                }));

PYBIND_REGISTER(InvertOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::InvertOperation, TensorOperation, std::shared_ptr<vision::InvertOperation>>(
                    *m, "InvertOperation")
                    .def(py::init([](const std::string &device_target) {
                      auto invert = std::make_shared<vision::InvertOperation>(device_target);
                      THROW_IF_ERROR(invert->ValidateParams());
                      return invert;
                    }));
                }));

PYBIND_REGISTER(
  MixUpBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MixUpBatchOperation, TensorOperation, std::shared_ptr<vision::MixUpBatchOperation>>(
      *m, "MixUpBatchOperation")
      .def(py::init([](float alpha) {
        auto mix_up_batch = std::make_shared<vision::MixUpBatchOperation>(alpha);
        THROW_IF_ERROR(mix_up_batch->ValidateParams());
        return mix_up_batch;
      }));
  }));

PYBIND_REGISTER(
  NormalizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizeOperation, TensorOperation, std::shared_ptr<vision::NormalizeOperation>>(
      *m, "NormalizeOperation")
      .def(py::init(
        [](const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc, std::string device_target) {
          auto normalize = std::make_shared<vision::NormalizeOperation>(mean, std, is_hwc, device_target);
          THROW_IF_ERROR(normalize->ValidateParams());
          return normalize;
        }));
  }));

PYBIND_REGISTER(
  NormalizePadOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizePadOperation, TensorOperation, std::shared_ptr<vision::NormalizePadOperation>>(
      *m, "NormalizePadOperation")
      .def(py::init(
        [](const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype, bool is_hwc) {
          auto normalize_pad = std::make_shared<vision::NormalizePadOperation>(mean, std, dtype, is_hwc);
          THROW_IF_ERROR(normalize_pad->ValidateParams());
          return normalize_pad;
        }));
  }));

PYBIND_REGISTER(PadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::PadOperation, TensorOperation, std::shared_ptr<vision::PadOperation>>(
                    *m, "PadOperation")
                    .def(py::init([](const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value,
                                     BorderType padding_mode, const std::string &device_target) {
                      auto pad =
                        std::make_shared<vision::PadOperation>(padding, fill_value, padding_mode, device_target);
                      THROW_IF_ERROR(pad->ValidateParams());
                      return pad;
                    }));
                }));

PYBIND_REGISTER(
  PadToSizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::PadToSizeOperation, TensorOperation, std::shared_ptr<vision::PadToSizeOperation>>(
      *m, "PadToSizeOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                       const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
        auto pad_to_size = std::make_shared<vision::PadToSizeOperation>(size, offset, fill_value, padding_mode);
        THROW_IF_ERROR(pad_to_size->ValidateParams());
        return pad_to_size;
      }));
  }));

PYBIND_REGISTER(
  PerspectiveOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::PerspectiveOperation, TensorOperation, std::shared_ptr<vision::PerspectiveOperation>>(
      *m, "PerspectiveOperation", "Tensor operation to apply perspective transformations on an image.")
      .def(py::init([](const std::vector<std::vector<int32_t>> &start_points,
                       const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation,
                       const std::string &device_target) {
        auto perspective =
          std::make_shared<vision::PerspectiveOperation>(start_points, end_points, interpolation, device_target);
        THROW_IF_ERROR(perspective->ValidateParams());
        return perspective;
      }));
  }));

PYBIND_REGISTER(
  PosterizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::PosterizeOperation, TensorOperation, std::shared_ptr<vision::PosterizeOperation>>(
      *m, "PosterizeOperation")
      .def(py::init([](uint8_t bits, const std::string &device_target) {
        auto posterize = std::make_shared<vision::PosterizeOperation>(bits, device_target);
        THROW_IF_ERROR(posterize->ValidateParams());
        return posterize;
      }));
  }));

PYBIND_REGISTER(
  RandAugmentOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandAugmentOperation, TensorOperation, std::shared_ptr<vision::RandAugmentOperation>>(
      *m, "RandAugmentOperation")
      .def(py::init([](int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins, InterpolationMode interpolation,
                       const std::vector<uint8_t> &fill_value) {
        auto rand_augment = std::make_shared<vision::RandAugmentOperation>(num_ops, magnitude, num_magnitude_bins,
                                                                           interpolation, fill_value);
        THROW_IF_ERROR(rand_augment->ValidateParams());
        return rand_augment;
      }));
  }));

PYBIND_REGISTER(RandomAdjustSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomAdjustSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomAdjustSharpnessOperation>>(
                    *m, "RandomAdjustSharpnessOperation")
                    .def(py::init([](float degree, float prob) {
                      auto random_adjust_sharpness =
                        std::make_shared<vision::RandomAdjustSharpnessOperation>(degree, prob);
                      THROW_IF_ERROR(random_adjust_sharpness->ValidateParams());
                      return random_adjust_sharpness;
                    }));
                }));

PYBIND_REGISTER(
  RandomAffineOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomAffineOperation, TensorOperation, std::shared_ptr<vision::RandomAffineOperation>>(
      *m, "RandomAffineOperation", "Tensor operation to apply random affine transformations on an image.")
      .def(py::init([](const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                       const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                       InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
        auto random_affine = std::make_shared<vision::RandomAffineOperation>(degrees, translate_range, scale_range,
                                                                             shear_ranges, interpolation, fill_value);
        THROW_IF_ERROR(random_affine->ValidateParams());
        return random_affine;
      }));
  }));

PYBIND_REGISTER(RandomAutoContrastOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomAutoContrastOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomAutoContrastOperation>>(*m,
                                                                                         "RandomAutoContrastOperation")
                    .def(py::init([](float cutoff, const std::vector<uint32_t> &ignore, float prob) {
                      auto random_auto_contrast =
                        std::make_shared<vision::RandomAutoContrastOperation>(cutoff, ignore, prob);
                      THROW_IF_ERROR(random_auto_contrast->ValidateParams());
                      return random_auto_contrast;
                    }));
                }));

PYBIND_REGISTER(RandomColorAdjustOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomColorAdjustOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomColorAdjustOperation>>(*m,
                                                                                        "RandomColorAdjustOperation")
                    .def(py::init([](const std::vector<float> &brightness, const std::vector<float> &contrast,
                                     const std::vector<float> &saturation, const std::vector<float> &hue) {
                      auto random_color_adjust =
                        std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
                      THROW_IF_ERROR(random_color_adjust->ValidateParams());
                      return random_color_adjust;
                    }));
                }));

PYBIND_REGISTER(
  RandomColorOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomColorOperation, TensorOperation, std::shared_ptr<vision::RandomColorOperation>>(
      *m, "RandomColorOperation")
      .def(py::init([](float t_lb, float t_ub) {
        auto random_color = std::make_shared<vision::RandomColorOperation>(t_lb, t_ub);
        THROW_IF_ERROR(random_color->ValidateParams());
        return random_color;
      }));
  }));

PYBIND_REGISTER(
  RandomCropDecodeResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropDecodeResizeOperation, TensorOperation,
                     std::shared_ptr<vision::RandomCropDecodeResizeOperation>>(*m, "RandomCropDecodeResizeOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_crop_decode_resize =
          std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_crop_decode_resize->ValidateParams());
        return random_crop_decode_resize;
      }));
  }));

PYBIND_REGISTER(
  RandomCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropOperation, TensorOperation, std::shared_ptr<vision::RandomCropOperation>>(
      *m, "RandomCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &padding, bool pad_if_needed,
                       const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
        auto random_crop =
          std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
        THROW_IF_ERROR(random_crop->ValidateParams());
        return random_crop;
      }));
  }));

PYBIND_REGISTER(RandomCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomCropWithBBoxOperation>>(*m,
                                                                                         "RandomCropWithBBoxOperation")
                    .def(
                      py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &padding,
                                  bool pad_if_needed, const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
                        auto random_crop_with_bbox = std::make_shared<vision::RandomCropWithBBoxOperation>(
                          size, padding, pad_if_needed, fill_value, padding_mode);
                        THROW_IF_ERROR(random_crop_with_bbox->ValidateParams());
                        return random_crop_with_bbox;
                      }));
                }));

PYBIND_REGISTER(RandomEqualizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomEqualizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomEqualizeOperation>>(*m, "RandomEqualizeOperation")
                    .def(py::init([](float prob) {
                      auto random_equalize = std::make_shared<vision::RandomEqualizeOperation>(prob);
                      THROW_IF_ERROR(random_equalize->ValidateParams());
                      return random_equalize;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipOperation>>(
                    *m, "RandomHorizontalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip->ValidateParams());
                      return random_horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipWithBBoxOperation>>(
                    *m, "RandomHorizontalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip_with_bbox =
                        std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip_with_bbox->ValidateParams());
                      return random_horizontal_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomInvertOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomInvertOperation, TensorOperation, std::shared_ptr<vision::RandomInvertOperation>>(
      *m, "RandomInvertOperation")
      .def(py::init([](float prob) {
        auto random_invert = std::make_shared<vision::RandomInvertOperation>(prob);
        THROW_IF_ERROR(random_invert->ValidateParams());
        return random_invert;
      }));
  }));

PYBIND_REGISTER(RandomLightingOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomLightingOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomLightingOperation>>(*m, "RandomLightingOperation")
                    .def(py::init([](float alpha) {
                      auto random_lighting = std::make_shared<vision::RandomLightingOperation>(alpha);
                      THROW_IF_ERROR(random_lighting->ValidateParams());
                      return random_lighting;
                    }));
                }));

PYBIND_REGISTER(RandomPosterizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomPosterizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomPosterizeOperation>>(*m, "RandomPosterizeOperation")
                    .def(py::init([](const std::vector<uint8_t> &bit_range) {
                      auto random_posterize = std::make_shared<vision::RandomPosterizeOperation>(bit_range);
                      THROW_IF_ERROR(random_posterize->ValidateParams());
                      return random_posterize;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizedCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizedCropOperation, TensorOperation,
                     std::shared_ptr<vision::RandomResizedCropOperation>>(*m, "RandomResizedCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_resized_crop =
          std::make_shared<vision::RandomResizedCropOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_resized_crop->ValidateParams());
        return random_resized_crop;
      }));
  }));

PYBIND_REGISTER(RandomResizedCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizedCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizedCropWithBBoxOperation>>(
                    *m, "RandomResizedCropWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                                     const std::vector<float> &ratio, InterpolationMode interpolation,
                                     int32_t max_attempts) {
                      auto random_resized_crop_with_bbox = std::make_shared<vision::RandomResizedCropWithBBoxOperation>(
                        size, scale, ratio, interpolation, max_attempts);
                      THROW_IF_ERROR(random_resized_crop_with_bbox->ValidateParams());
                      return random_resized_crop_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizeOperation, TensorOperation, std::shared_ptr<vision::RandomResizeOperation>>(
      *m, "RandomResizeOperation")
      .def(py::init([](const std::vector<int32_t> &size) {
        auto random_resize = std::make_shared<vision::RandomResizeOperation>(size);
        THROW_IF_ERROR(random_resize->ValidateParams());
        return random_resize;
      }));
  }));

PYBIND_REGISTER(RandomResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizeWithBBoxOperation>>(
                    *m, "RandomResizeWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size) {
                      auto random_resize_with_bbox = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
                      THROW_IF_ERROR(random_resize_with_bbox->ValidateParams());
                      return random_resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomRotationOperation, 1, ([](const py::module *m) {
    (void)
      py::class_<vision::RandomRotationOperation, TensorOperation, std::shared_ptr<vision::RandomRotationOperation>>(
        *m, "RandomRotationOperation")
        .def(py::init([](const std::vector<float> &degrees, InterpolationMode interpolation_mode, bool expand,
                         const std::vector<float> &center, const std::vector<uint8_t> &fill_value) {
          auto random_rotation =
            std::make_shared<vision::RandomRotationOperation>(degrees, interpolation_mode, expand, center, fill_value);
          THROW_IF_ERROR(random_rotation->ValidateParams());
          return random_rotation;
        }));
  }));

PYBIND_REGISTER(
  RandomSelectSubpolicyOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomSelectSubpolicyOperation, TensorOperation,
                     std::shared_ptr<vision::RandomSelectSubpolicyOperation>>(*m, "RandomSelectSubpolicyOperation")
      .def(py::init([](const py::list &py_policy) {
        std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> cpp_policy;
        for (auto &py_sub : py_policy) {
          cpp_policy.push_back({});
          for (auto handle : py_sub.cast<py::list>()) {
            py::tuple tp = handle.cast<py::tuple>();
            if (tp.is_none() || tp.size() != 2) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "Each tuple in subpolicy should be (op, prob)."));
            }
            std::shared_ptr<TensorOperation> t_op;
            if (py::isinstance<TensorOperation>(tp[0])) {
              t_op = (tp[0]).cast<std::shared_ptr<TensorOperation>>();
            } else if (py::isinstance<TensorOp>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>((tp[0]).cast<std::shared_ptr<TensorOp>>());
            } else if (py::isinstance<py::function>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>(
                std::make_shared<PyFuncOp>((tp[0]).cast<py::function>()));
            } else {
              THROW_IF_ERROR(
                Status(StatusCode::kMDUnexpectedError, "op is neither a tensorOp, tensorOperation nor a pyfunc."));
            }
            double prob = (tp[1]).cast<py::float_>();
            if (prob < 0 || prob > 1) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "prob needs to be with [0,1]."));
            }
            cpp_policy.back().emplace_back(std::make_pair(t_op, prob));
          }
        }
        auto random_select_subpolicy = std::make_shared<vision::RandomSelectSubpolicyOperation>(cpp_policy);
        THROW_IF_ERROR(random_select_subpolicy->ValidateParams());
        return random_select_subpolicy;
      }));
  }));

PYBIND_REGISTER(RandomSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSharpnessOperation>>(*m, "RandomSharpnessOperation")
                    .def(py::init([](const std::vector<float> &degrees) {
                      auto random_sharpness = std::make_shared<vision::RandomSharpnessOperation>(degrees);
                      THROW_IF_ERROR(random_sharpness->ValidateParams());
                      return random_sharpness;
                    }));
                }));

PYBIND_REGISTER(RandomSolarizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSolarizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSolarizeOperation>>(*m, "RandomSolarizeOperation")
                    .def(py::init([](const std::vector<uint8_t> &threshold) {
                      auto random_solarize = std::make_shared<vision::RandomSolarizeOperation>(threshold);
                      THROW_IF_ERROR(random_solarize->ValidateParams());
                      return random_solarize;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipOperation>>(*m,
                                                                                         "RandomVerticalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip = std::make_shared<vision::RandomVerticalFlipOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip->ValidateParams());
                      return random_vertical_flip;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipWithBBoxOperation>>(
                    *m, "RandomVerticalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip_with_bbox =
                        std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip_with_bbox->ValidateParams());
                      return random_vertical_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(ReadFileOperation, 1, ([](py::module *m) {
                  (void)m->def("read_file", ([](const std::string &filename) {
                                 std::shared_ptr<Tensor> output;
                                 THROW_IF_ERROR(ReadFile(filename, &output));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(ReadImageOperation, 1, ([](py::module *m) {
                  (void)m->def("read_image", ([](const std::string &filename, ImageReadMode mode) {
                                 std::shared_ptr<Tensor> output;
                                 THROW_IF_ERROR(mindspore::dataset::ReadImage(filename, &output, mode));
                                 return output;
                               }));
                }));

#ifdef ENABLE_FFMPEG
PYBIND_REGISTER(ReadVideo, 1, ([](py::module *m) {
                  (void)m->def(
                    "read_video",
                    ([](const std::string &filename, float start_pts, float end_pts, const std::string &pts_unit) {
                      std::shared_ptr<Tensor> video_output;
                      std::shared_ptr<Tensor> audio_output;
                      std::map<std::string, std::string> metadata_output;
                      THROW_IF_ERROR(mindspore::dataset::ReadVideo(filename, &video_output, &audio_output,
                                                                   &metadata_output, start_pts, end_pts, pts_unit));
                      return std::make_tuple(video_output, audio_output, metadata_output);
                    }));
                }));

PYBIND_REGISTER(ReadVideoTimestampsOperation, 1, ([](py::module *m) {
                  (void)m->def("read_video_timestamps", ([](const std::string &filename, const std::string &pts_unit) {
                                 std::vector<int64_t> pts_int64_vector;
                                 float video_fps;
                                 float time_base;
                                 THROW_IF_ERROR(mindspore::dataset::ReadVideoTimestamps(
                                   filename, &pts_int64_vector, &video_fps, &time_base, pts_unit));
                                 return std::make_tuple(pts_int64_vector, video_fps, time_base);
                               }));
                }));
#endif

PYBIND_REGISTER(RescaleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RescaleOperation, TensorOperation, std::shared_ptr<vision::RescaleOperation>>(
                      *m, "RescaleOperation")
                      .def(py::init([](float rescale, float shift) {
                        auto rescale_op = std::make_shared<vision::RescaleOperation>(rescale, shift);
                        THROW_IF_ERROR(rescale_op->ValidateParams());
                        return rescale_op;
                      }));
                }));

PYBIND_REGISTER(ResizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeOperation, TensorOperation, std::shared_ptr<vision::ResizeOperation>>(
                    *m, "ResizeOperation")
                    .def(py::init([](const std::vector<int32_t> &size, InterpolationMode interpolation_mode,
                                     std::string device_target) {
                      auto resize = std::make_shared<vision::ResizeOperation>(size, interpolation_mode, device_target);
                      THROW_IF_ERROR(resize->ValidateParams());
                      return resize;
                    }));
                }));

PYBIND_REGISTER(ResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::ResizeWithBBoxOperation>>(*m, "ResizeWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size, InterpolationMode interpolation_mode) {
                      auto resize_with_bbox =
                        std::make_shared<vision::ResizeWithBBoxOperation>(size, interpolation_mode);
                      THROW_IF_ERROR(resize_with_bbox->ValidateParams());
                      return resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  ResizedCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::ResizedCropOperation, TensorOperation, std::shared_ptr<vision::ResizedCropOperation>>(
      *m, "ResizedCropOperation")
      .def(py::init([](int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
                       InterpolationMode interpolation, const std::string &device_target) {
        auto resized_crop =
          std::make_shared<vision::ResizedCropOperation>(top, left, height, width, size, interpolation, device_target);
        THROW_IF_ERROR(resized_crop->ValidateParams());
        return resized_crop;
      }));
  }));

PYBIND_REGISTER(RgbToBgrOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RgbToBgrOperation, TensorOperation, std::shared_ptr<vision::RgbToBgrOperation>>(
                      *m, "RgbToBgrOperation")
                      .def(py::init([]() {
                        auto rgb2bgr = std::make_shared<vision::RgbToBgrOperation>();
                        THROW_IF_ERROR(rgb2bgr->ValidateParams());
                        return rgb2bgr;
                      }));
                }));

PYBIND_REGISTER(RotateOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RotateOperation, TensorOperation, std::shared_ptr<vision::RotateOperation>>(
                    *m, "RotateOperation")
                    .def(py::init([](float degrees, InterpolationMode resample, bool expand,
                                     const std::vector<float> &center, const std::vector<uint8_t> &fill_value,
                                     const std::string &device_target) {
                      auto rotate = std::make_shared<vision::RotateOperation>(degrees, resample, expand, center,
                                                                              fill_value, device_target);
                      THROW_IF_ERROR(rotate->ValidateParams());
                      return rotate;
                    }));
                }));

PYBIND_REGISTER(
  SlicePatchesOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::SlicePatchesOperation, TensorOperation, std::shared_ptr<vision::SlicePatchesOperation>>(
      *m, "SlicePatchesOperation")
      .def(py::init([](int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value) {
        auto slice_patches =
          std::make_shared<vision::SlicePatchesOperation>(num_height, num_width, slice_mode, fill_value);
        THROW_IF_ERROR(slice_patches->ValidateParams());
        return slice_patches;
      }));
  }));

PYBIND_REGISTER(SolarizeOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::SolarizeOperation, TensorOperation, std::shared_ptr<vision::SolarizeOperation>>(
                      *m, "SolarizeOperation")
                      .def(py::init([](const std::vector<float> &threshold, const std::string &device_target) {
                        auto solarize = std::make_shared<vision::SolarizeOperation>(threshold, device_target);
                        THROW_IF_ERROR(solarize->ValidateParams());
                        return solarize;
                      }));
                }));

PYBIND_REGISTER(ToTensorOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::ToTensorOperation, TensorOperation, std::shared_ptr<vision::ToTensorOperation>>(
                      *m, "ToTensorOperation")
                      .def(py::init([](const std::string &output_type) {
                        auto totensor = std::make_shared<vision::ToTensorOperation>(output_type);
                        THROW_IF_ERROR(totensor->ValidateParams());
                        return totensor;
                      }));
                }));

PYBIND_REGISTER(TrivialAugmentWideOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::TrivialAugmentWideOperation, TensorOperation,
                                   std::shared_ptr<vision::TrivialAugmentWideOperation>>(*m,
                                                                                         "TrivialAugmentWideOperation")
                    .def(py::init([](int32_t num_magnitude_bins, InterpolationMode interpolation,
                                     const std::vector<uint8_t> &fill_value) {
                      auto auto_augment = std::make_shared<vision::TrivialAugmentWideOperation>(
                        num_magnitude_bins, interpolation, fill_value);
                      THROW_IF_ERROR(auto_augment->ValidateParams());
                      return auto_augment;
                    }));
                }));

PYBIND_REGISTER(
  UniformAugOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::UniformAugOperation, TensorOperation, std::shared_ptr<vision::UniformAugOperation>>(
      *m, "UniformAugOperation")
      .def(py::init([](const py::list &transforms, int32_t num_ops) {
        auto uniform_aug =
          std::make_shared<vision::UniformAugOperation>(std::move(toTensorOperations(transforms)), num_ops);
        THROW_IF_ERROR(uniform_aug->ValidateParams());
        return uniform_aug;
      }));
  }));

PYBIND_REGISTER(
  VerticalFlipOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::VerticalFlipOperation, TensorOperation, std::shared_ptr<vision::VerticalFlipOperation>>(
      *m, "VerticalFlipOperation")
      .def(py::init([](const std::string &device_target) {
        auto vertical_flip = std::make_shared<vision::VerticalFlipOperation>(device_target);
        THROW_IF_ERROR(vertical_flip->ValidateParams());
        return vertical_flip;
      }));
  }));

PYBIND_REGISTER(WriteFileOperation, 1, ([](py::module *m) {
                  (void)m->def("write_file", ([](const std::string &filename, const std::shared_ptr<Tensor> &data) {
                                 THROW_IF_ERROR(WriteFile(filename, data));
                               }));
                }));

PYBIND_REGISTER(WriteJPEGOperation, 1, ([](py::module *m) {
                  (void)m->def("write_jpeg",
                               ([](const std::string &filename, const std::shared_ptr<Tensor> &image, int quality) {
                                 THROW_IF_ERROR(WriteJpeg(filename, image, quality));
                               }));
                }));

PYBIND_REGISTER(WritePNGOperation, 1, ([](py::module *m) {
                  (void)m->def("write_png", ([](const std::string &filename, const std::shared_ptr<Tensor> &image,
                                                int compression_level) {
                                 THROW_IF_ERROR(WritePng(filename, image, compression_level));
                               }));
                }));

#if defined(ENABLE_D)
PYBIND_REGISTER(DvppCodec, 1, ([](py::module *m) {
                  (void)m->def(
                    "dvpp_sys_init", []() { THROW_IF_ERROR(AclAdapter::GetInstance().DvppSysInit()); },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "dvpp_sys_exit", []() { THROW_IF_ERROR(AclAdapter::GetInstance().DvppSysExit()); },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_create_chn",
                    [](int ptype) {
                      int64_t chnl;
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecCreateChnl(ptype, &chnl));
                      return chnl;
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_start_get_frame",
                    [](int chnId, int totalFrame) {
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecStartGetFrame(chnId, totalFrame));
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_send_stream",
                    [](int chnId, const std::shared_ptr<Tensor> &input, int64_t outFormat, bool display,
                       std::shared_ptr<DeviceBuffer> *out) {
                      Status status =
                        AclAdapter::GetInstance().DvppVdecSendStream(chnId, input, outFormat, display, out);
                      if (status.IsError()) {
                        MS_LOG(WARNING) << status.ToString();
                      }
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_stop_get_frame",
                    [](int chnId, int totalFrame) {
                      std::shared_ptr<DeviceBuffer> output;
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecStopGetFrame(chnId, totalFrame, &output));
                      return output;
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_destroy_chnl",
                    [](int chnId) {
                      Status status = AclAdapter::GetInstance().DvppVdecDestroyChnl(chnId);
                      if (status.IsError()) {
                        MS_LOG(WARNING) << status.ToString();
                      }
                    },
                    py::call_guard<py::gil_scoped_release>());
                }));
#endif

PYBIND_REGISTER(PyAV, 0, ([](py::module *m) {
                  (void)m->def(
                    "pyav_open",
                    [](const std::string &file) {
                      auto container = std::make_shared<Container>(file);
                      THROW_IF_ERROR(container->Init());
                      return container;
                    },
                    py::call_guard<py::gil_scoped_release>());
                }));

PYBIND_REGISTER(CodecContext, 0, ([](const py::module *m) {
                  (void)py::class_<CodecContext, std::shared_ptr<CodecContext>>(*m, "CodecContext")
                    .def_property_readonly("extradata", ([](CodecContext &codec_context) {
                                             auto extradata = codec_context.GetExtradata();
                                             return py::bytes(extradata);
                                           }));
                }));

PYBIND_REGISTER(Stream, 0, ([](const py::module *m) {
                  (void)py::class_<Stream, std::shared_ptr<Stream>>(*m, "Stream")
                    .def_property_readonly("type", &Stream::GetType)
                    .def_property_readonly("time_base", ([](Stream &stream) {
                                             auto time_base = stream.GetTimeBase();
                                             py::object Fraction = py::module::import("fractions").attr("Fraction");
                                             return Fraction(time_base->num, time_base->den);
                                           }))
                    .def_property_readonly("start_time", ([](Stream &stream) -> py::object {
                                             auto start_time = stream.GetStartTime();
                                             if (start_time == -1) {
                                               return py::none();
                                             }
                                             return py::int_(start_time);
                                           }))
                    .def_property_readonly("bit_rate", ([](Stream &stream) -> py::object {
                                             auto bit_rate = stream.GetBitRate();
                                             if (bit_rate == -1) {
                                               return py::none();
                                             }
                                             return py::int_(bit_rate);
                                           }))
                    .def_property_readonly("duration", ([](Stream &stream) -> py::object {
                                             auto duration = stream.GetDuration();
                                             if (duration == -1) {
                                               return py::none();
                                             }
                                             return py::int_(duration);
                                           }))
                    .def_property_readonly("frames", &Stream::GetFrames)
                    .def_property_readonly("flags", &Stream::GetFlags)
                    .def_property_readonly("codec_context", &Stream::GetCodecContext);
                }));

PYBIND_REGISTER(Packet, 0, ([](const py::module *m) {
                  (void)py::class_<Packet, std::shared_ptr<Packet>>(*m, "Packet")
                    .def_property_readonly("is_keyframe", &Packet::IsKeyFrame)
                    .def_property_readonly("pts", ([](Packet &packet) -> py::object {
                                             auto pts = packet.GetPTS();
                                             if (pts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(pts);
                                           }))
                    .def_property_readonly("dts", ([](Packet &packet) -> py::object {
                                             auto dts = packet.GetDTS();
                                             if (dts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(dts);
                                           }));
                }));

PYBIND_REGISTER(Frame, 0, ([](const py::module *m) {
                  (void)py::class_<Frame, std::shared_ptr<Frame>>(*m, "Frame")
                    .def_property_readonly("pts", ([](Frame &frame) -> py::object {
                                             auto pts = frame.GetPTS();
                                             if (pts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(pts);
                                           }));
                }));

PYBIND_REGISTER(AudioFrame, 1, ([](const py::module *m) {
                  (void)py::class_<AudioFrame, Frame, std::shared_ptr<AudioFrame>>(*m, "AudioFrame")
                    .def("to_ndarray", [](AudioFrame &audio_frame) { return audio_frame.ToNumpy(); });
                }));

PYBIND_REGISTER(VideoStream, 1, ([](const py::module *m) {
                  (void)py::class_<VideoStream, Stream, std::shared_ptr<VideoStream>>(*m, "VideoStream")
                    .def_property_readonly("name", &VideoStream::GetName)
                    .def_property_readonly("average_rate", ([](VideoStream &video_stream) {
                                             auto average_rate = video_stream.GetAverageRate();
                                             py::object Fraction = py::module::import("fractions").attr("Fraction");
                                             return Fraction(average_rate->num, average_rate->den);
                                           }))
                    .def_property_readonly("width", &VideoStream::GetWidth)
                    .def_property_readonly("height", &VideoStream::GetHeight);
                }));

PYBIND_REGISTER(AudioStream, 1, ([](const py::module *m) {
                  (void)py::class_<AudioStream, Stream, std::shared_ptr<AudioStream>>(*m, "AudioStream")
                    .def_property_readonly("rate", &AudioStream::GetRate);
                }));

PYBIND_REGISTER(StreamContainer, 2, ([](const py::module *m) {
                  (void)py::class_<StreamContainer, std::shared_ptr<StreamContainer>>(*m, "StreamContainer")
                    .def_property_readonly("video", &StreamContainer::GetVideo)
                    .def_property_readonly("audio", &StreamContainer::GetAudio);
                }));

PYBIND_REGISTER(Container, 3, ([](const py::module *m) {
                  (void)py::class_<Container, std::shared_ptr<Container>>(*m, "AVContainer")
                    .def(
                      "__enter__", [](Container &container) -> Container & { return container; },
                      py::return_value_policy::reference_internal)
                    .def(
                      "__exit__", [](Container &container, py::object, py::object, py::object) { container.Close(); },
                      py::call_guard<py::gil_scoped_release>())
                    .def_property_readonly("streams", &Container::GetStreams)
                    .def(
                      "demux",
                      [](Container &container, const std::shared_ptr<Stream> &stream) {
                        std::vector<std::shared_ptr<Packet>> packets;
                        THROW_IF_ERROR(container.Demux(stream, &packets));
                        return packets;
                      },
                      py::call_guard<py::gil_scoped_release>())
                    .def(
                      "decode",
                      [](Container &container, int streams, int video, int audio) {
                        auto stream = container.GetStreams().Get(streams, video, audio);
                        std::vector<std::shared_ptr<Frame>> frames;
                        THROW_IF_ERROR(container.Decode(stream, &frames));
                        return frames;
                      },
                      py::arg("streams") = -1, py::arg("video") = -1, py::arg("audio") = -1,
                      py::call_guard<py::gil_scoped_release>())
                    .def(
                      "seek",
                      [](Container &container, int64_t offset, bool backward, bool any_frame,
                         const std::shared_ptr<Stream> &stream) {
                        THROW_IF_ERROR(container.Seek(offset, backward, any_frame, stream));
                      },
                      py::arg("offset"), py::kw_only(), py::arg("backward"), py::arg("any_frame"), py::arg("stream"),
                      py::call_guard<py::gil_scoped_release>());
                }));
}  // namespace dataset
}  // namespace mindspore
