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

#include "minddata/dataset/vision/kernels/pyav/frame.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mindspore::dataset {
std::unordered_map<std::string, std::string> format_dtypes = {
  {"dbl", "f8"},  {"dblp", "f8"}, {"flt", "f4"},  {"fltp", "f4"}, {"s16", "i2"},
  {"s16p", "i2"}, {"s32", "i4"},  {"s32p", "i4"}, {"u8", "u1"},   {"u8p", "u1"},
};

Frame::Frame() : frame_(av_frame_alloc()), time_base_(nullptr) {}

Frame::~Frame() { av_frame_free(&frame_); }

void Frame::SetTimeBase(AVRational *time_base) { time_base_ = time_base; }

int Frame::GetPTS() {
  if (frame_->pts == AV_NOPTS_VALUE) {
    return -1;
  }
  return frame_->pts;
}

AVFrame *Frame::GetAVFrame() { return frame_; }

Status Frame::InitUserAttributes() { return Status::OK(); }

VideoFrame::VideoFrame() = default;

Status AllocVideoFrame(std::shared_ptr<Frame> *frame) {
  RETURN_UNEXPECTED_IF_NULL(frame);
  *frame = std::make_shared<VideoFrame>();
  return Status::OK();
}

AudioFrame::AudioFrame() = default;

Status AudioFrame::InitUserAttributes() {
  layout_ = frame_->ch_layout;
  format_ = static_cast<AVSampleFormat>(frame_->format);
  return Status::OK();
}

std::vector<py::array> AudioFrame::ToNumpy() {
  auto dtype = py::dtype(format_dtypes[av_get_sample_fmt_name(format_)]);

  std::vector<py::array> res;
  if (av_sample_fmt_is_planar(format_)) {
    for (auto channel = 0; channel < layout_.nb_channels; ++channel) {
      py::array arr = py::array(dtype, {frame_->nb_samples}, {dtype.itemsize()}, frame_->extended_data[channel]);
      res.emplace_back(arr);
    }
  } else {
    py::array arr =
      py::array(dtype, {frame_->nb_samples * layout_.nb_channels}, {dtype.itemsize()}, frame_->extended_data[0]);
    res.emplace_back(arr);
  }

  return res;
}

Status AllocAudioFrame(std::shared_ptr<Frame> *frame) {
  RETURN_UNEXPECTED_IF_NULL(frame);
  *frame = std::make_shared<AudioFrame>();
  return Status::OK();
}
}  // namespace mindspore::dataset
