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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_FORMAT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_FORMAT_H_

extern "C" {
#include <libavutil/pixdesc.h>
}

#include <memory>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class VideoFormat {
 public:
  VideoFormat() = default;

  void Init(AVPixelFormat pixel_format, uint32_t width, uint32_t height);

 private:
  AVPixelFormat pixel_format_;
  const AVPixFmtDescriptor *pix_fmt_descriptor_;
  uint32_t width_;
  uint32_t height_;
};

Status GetVideoFormat(AVPixelFormat pixel_format, uint32_t width, uint32_t height,
                      std::shared_ptr<VideoFormat> *output_video_format);
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_FORMAT_H_
