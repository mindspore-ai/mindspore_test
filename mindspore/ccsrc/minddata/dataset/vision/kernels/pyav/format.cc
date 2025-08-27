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

#include "minddata/dataset/vision/kernels/pyav/format.h"

#include <memory>
#include <string>
#include <vector>

namespace mindspore::dataset {
void VideoFormat::Init(AVPixelFormat pixel_format, uint32_t width, uint32_t height) {
  pixel_format_ = pixel_format;
  pix_fmt_descriptor_ = av_pix_fmt_desc_get(pixel_format);
  width_ = width;
  height_ = height;
}

Status GetVideoFormat(AVPixelFormat pixel_format, uint32_t width, uint32_t height,
                      std::shared_ptr<VideoFormat> *output_video_format) {
  RETURN_UNEXPECTED_IF_NULL(output_video_format);

  if (pixel_format == AV_PIX_FMT_NONE) {
    RETURN_STATUS_UNEXPECTED("AVPixelFormat is none.");
  }
  *output_video_format = std::make_shared<VideoFormat>();
  (*output_video_format)->Init(pixel_format, width, height);

  return Status::OK();
}
}  // namespace mindspore::dataset
