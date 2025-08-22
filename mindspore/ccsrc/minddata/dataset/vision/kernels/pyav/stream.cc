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

#include "minddata/dataset/vision/kernels/pyav/stream.h"

#include <memory>
#include <vector>

#include "minddata/dataset/vision/kernels/pyav/container.h"
#include "minddata/dataset/vision/kernels/pyav/context.h"
#include "minddata/dataset/vision/kernels/pyav/frame.h"
#include "minddata/dataset/vision/kernels/pyav/packet.h"

namespace mindspore::dataset {
void Stream::Init(std::shared_ptr<Container> container, AVStream *av_stream,
                  std::shared_ptr<CodecContext> codec_context) {
  container_ = container;
  stream_ = av_stream;
  codec_context_ = codec_context;
  if (codec_context_ != nullptr) {
    codec_context_->SetStreamIndex(av_stream->index);
  }
}

Status Stream::Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) {
  RETURN_STATUS_UNEXPECTED("Decode method is not supported for Stream.");
  return Status::OK();
}

AVRational *Stream::GetTimeBase() { return &(stream_->time_base); }

int Stream::GetStartTime() {
  if (stream_->start_time != AV_NOPTS_VALUE) {
    return stream_->start_time;
  }
  return -1;
}

int Stream::GetBitRate() const { return codec_context_->GetBitRate(); }

int Stream::GetFlags() const { return codec_context_->GetFlags(); }

int Stream::GetFrames() const { return stream_->nb_frames; }

int Stream::GetDuration() const {
  if (stream_->duration != AV_NOPTS_VALUE) {
    return stream_->duration;
  }
  return -1;
}

const std::shared_ptr<CodecContext> &Stream::GetCodecContext() const { return codec_context_; }

const char *Stream::GetType() const { return av_get_media_type_string(stream_->codecpar->codec_type); }

Status VideoStream::Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) {
  return codec_context_->Decode(packet, frames);
}

AVRational *VideoStream::GetAverageRate() const { return &(stream_->avg_frame_rate); }

const char *VideoStream::GetName() const { return codec_context_->GetName(); }

int VideoStream::GetWidth() const { return std::static_pointer_cast<VideoCodecContext>(codec_context_)->GetWidth(); }

int VideoStream::GetHeight() const { return std::static_pointer_cast<VideoCodecContext>(codec_context_)->GetHeight(); }

Status AudioStream::Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) {
  return codec_context_->Decode(packet, frames);
}

int AudioStream::GetRate() { return std::static_pointer_cast<AudioCodecContext>(codec_context_)->GetRate(); }

Status WrapStream(std::shared_ptr<Container> container, AVStream *av_stream,
                  std::shared_ptr<CodecContext> codec_context, std::shared_ptr<Stream> *out_stream) {
  RETURN_UNEXPECTED_IF_NULL(out_stream);
  CHECK_FAIL_RETURN_UNEXPECTED(container->GetFormatContext()->streams[av_stream->index] == av_stream,
                               "Stream is not correct.");
  switch (av_stream->codecpar->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
      *out_stream = std::make_shared<VideoStream>();
      break;
    case AVMEDIA_TYPE_AUDIO:
      *out_stream = std::make_shared<AudioStream>();
      break;
    default:
      *out_stream = std::make_shared<Stream>();
  }
  (*out_stream)->Init(container, av_stream, codec_context);
  return Status::OK();
}
}  // namespace mindspore::dataset
