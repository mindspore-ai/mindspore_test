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

#include "minddata/dataset/vision/kernels/pyav/container.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/vision/kernels/pyav/context.h"
#include "minddata/dataset/vision/kernels/pyav/frame.h"
#include "minddata/dataset/vision/kernels/pyav/packet.h"
#include "minddata/dataset/vision/kernels/pyav/stream.h"

namespace mindspore::dataset {
Status StreamContainer::AddStream(std::shared_ptr<Stream> stream) {
  RETURN_UNEXPECTED_IF_NULL(stream);
  CHECK_FAIL_RETURN_UNEXPECTED(stream->GetAVStream()->index == streams_.size(), "Not all the streams have been added.");

  switch (stream->GetAVStream()->codecpar->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
      streams_.emplace_back(std::static_pointer_cast<VideoStream>(stream));
      video_.emplace_back(std::static_pointer_cast<VideoStream>(stream));
      break;
    case AVMEDIA_TYPE_AUDIO:
      streams_.emplace_back(std::static_pointer_cast<AudioStream>(stream));
      audio_.emplace_back(std::static_pointer_cast<AudioStream>(stream));
      break;
    default:
      streams_.emplace_back(stream);
      other_.emplace_back(stream);
  }

  return Status::OK();
}

std::shared_ptr<Stream> StreamContainer::operator[](size_t index) { return streams_[index]; }

std::shared_ptr<Stream> StreamContainer::Get(int streams, int video, int audio) const {
  if (streams != -1) {
    return streams_[streams];
  } else if (video != -1) {
    return video_[video];
  } else if (audio != -1) {
    return audio_[audio];
  }
  return streams_[0];
}

Container::Container(const std::string &file)
    : name_(file), input_was_opened_(false), format_context_(nullptr), streams_(StreamContainer()) {}

Status Container::Init() {
  format_context_ = avformat_alloc_context();
  format_context_->flags = static_cast<int>(static_cast<uint32_t>(format_context_->flags) | AVFMT_FLAG_GENPTS);
  format_context_->opaque = reinterpret_cast<void *>(this);

  CHECK_FAIL_RETURN_UNEXPECTED(avformat_open_input(&format_context_, name_.c_str(), nullptr, nullptr) == 0,
                               "Failed to open the file: " + name_);
  input_was_opened_ = true;

  if (avformat_find_stream_info(format_context_, nullptr) < 0) {
    avformat_close_input(&format_context_);
    RETURN_STATUS_UNEXPECTED("Failed to find stream info.");
  }

  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    AVStream *stream = format_context_->streams[i];
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    std::shared_ptr<CodecContext> py_codec_context;
    if (codec != nullptr) {
      AVCodecContext *codec_context = avcodec_alloc_context3(codec);
      CHECK_FAIL_RETURN_UNEXPECTED(avcodec_parameters_to_context(codec_context, stream->codecpar) >= 0,
                                   "Failed to call avcodec_parameters_to_context.");
      codec_context->pkt_timebase = stream->time_base;
      RETURN_IF_NOT_OK(WrapCodecContext(codec_context, codec, &py_codec_context));
    } else {
      py_codec_context = nullptr;
    }
    std::shared_ptr<Stream> out_stream;
    RETURN_IF_NOT_OK(WrapStream(shared_from_this(), stream, py_codec_context, &out_stream));
    RETURN_IF_NOT_OK(streams_.AddStream(out_stream));
  }

  return Status::OK();
}

void Container::Close() {
  streams_ = StreamContainer();
  if (input_was_opened_) {
    avformat_close_input(&format_context_);
    input_was_opened_ = false;
  }
}

Status Container::Demux(const std::shared_ptr<Stream> &stream, std::vector<std::shared_ptr<Packet>> *packets) {
  RETURN_UNEXPECTED_IF_NULL(packets);
  RETURN_IF_NOT_OK(AssertOpen());

  auto include_stream = std::vector<bool>(format_context_->nb_streams, false);
  uint32_t stream_index = stream->GetIndex();
  CHECK_FAIL_RETURN_UNEXPECTED(stream_index < format_context_->nb_streams,
                               "Stream index " + std::to_string(stream_index) + " is out of range.");
  include_stream[stream_index] = true;
  while (true) {
    auto packet = std::make_shared<Packet>();
    RETURN_UNEXPECTED_IF_NULL(packet->packet_);
    int ret = av_read_frame(format_context_, packet->packet_);
    if (ret < 0) {
      if (ret == AVERROR_EOF) {
        break;
      } else {
        RETURN_STATUS_UNEXPECTED("Failed to call av_read_frame, ret: " + std::to_string(ret));
      }
    }
    if (include_stream[packet->packet_->stream_index]) {
      if (packet->packet_->stream_index < streams_.Size()) {
        packet->stream_ = streams_[packet->packet_->stream_index];
        packet->time_base_ = packet->stream_->GetTimeBase();
        (*packets).emplace_back(packet);
      }
    }
  }
  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    if (include_stream[i]) {
      auto packet = std::make_shared<Packet>();
      packet->stream_ = streams_[i];
      packet->time_base_ = packet->stream_->GetTimeBase();
      (*packets).emplace_back(packet);
    }
  }
  return Status::OK();
}

Status Container::Decode(const std::shared_ptr<Stream> &stream, std::vector<std::shared_ptr<Frame>> *frames) {
  RETURN_UNEXPECTED_IF_NULL(frames);

  RETURN_IF_NOT_OK(AssertOpen());
  std::vector<std::shared_ptr<Packet>> packets;
  RETURN_IF_NOT_OK(Demux(stream, &packets));

  for (const auto &packet : packets) {
    std::vector<std::shared_ptr<Frame>> decoded_frames;
    RETURN_IF_NOT_OK(packet->Decode(&decoded_frames));
    (void)(*frames).insert((*frames).end(), decoded_frames.begin(), decoded_frames.end());
  }

  return Status::OK();
}

Status Container::Seek(int64_t offset, bool backward, bool any_frame, const std::shared_ptr<Stream> &stream) {
  RETURN_IF_NOT_OK(AssertOpen());

  uint32_t flags = 0;

  if (backward) {
    flags = flags | AVSEEK_FLAG_BACKWARD;
  }
  if (any_frame) {
    flags = flags | AVSEEK_FLAG_ANY;
  }

  int stream_index = -1;
  if (stream != nullptr) {
    stream_index = stream->GetIndex();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(av_seek_frame(format_context_, stream_index, offset, static_cast<int>(flags)) >= 0,
                               "Failed to call av_seek_frame.");

  RETURN_IF_NOT_OK(FlushBuffers());
  return Status::OK();
}

Status Container::FlushBuffers() {
  RETURN_IF_NOT_OK(AssertOpen());

  for (auto i = 0; i < streams_.Size(); ++i) {
    std::shared_ptr<CodecContext> codec_context = streams_[i]->GetCodecContext();
    if (codec_context != nullptr) {
      codec_context->FlushBuffers();
    }
  }
  return Status::OK();
}

Status Container::AssertOpen() const {
  CHECK_FAIL_RETURN_UNEXPECTED(format_context_ != nullptr, "Container is not open.");
  return Status::OK();
}

Container::~Container() {
  Close();
  avformat_free_context(format_context_);
}
}  // namespace mindspore::dataset
