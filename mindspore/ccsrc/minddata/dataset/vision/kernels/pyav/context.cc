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

#include "minddata/dataset/vision/kernels/pyav/context.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/vision/kernels/pyav/format.h"
#include "minddata/dataset/vision/kernels/pyav/frame.h"
#include "minddata/dataset/vision/kernels/pyav/packet.h"

namespace mindspore::dataset {
CodecContext::CodecContext() : stream_index_(-1), codec_context_(nullptr), codec_(nullptr), is_open_(false) {}

Status CodecContext::Init(AVCodecContext *codec_context, const AVCodec *codec) {
  codec_context_ = codec_context;
  if (codec_context_->codec != nullptr && codec != nullptr && codec_context_->codec != codec) {
    RETURN_STATUS_UNEXPECTED("Wrapping CodecContext with mismatched codec.");
  }
  codec_ = codec != nullptr ? codec : codec_context_->codec;
  return Status::OK();
}

Status CodecContext::Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) {
  RETURN_UNEXPECTED_IF_NULL(frames);
  CHECK_FAIL_RETURN_UNEXPECTED(codec_ != nullptr, "Cannot decode unknown codec.");

  RETURN_IF_NOT_OK(Open(false));

  std::vector<std::shared_ptr<Frame>> recv_frames;
  RETURN_IF_NOT_OK(SendPacketAndRecv(packet, &recv_frames));

  for (const auto &frame : recv_frames) {
    RETURN_IF_NOT_OK(SetupDecodedFrame(frame, packet));
    (*frames).emplace_back(frame);
  }
  return Status::OK();
}

Status CodecContext::SetupDecodedFrame(const std::shared_ptr<Frame> &frame, const std::shared_ptr<Packet> &packet) {
  if (packet != nullptr) {
    frame->SetTimeBase(packet->GetTimeBase());
  }
  return Status::OK();
}

Status CodecContext::Open(bool strict) {
  if (is_open_) {
    CHECK_FAIL_RETURN_UNEXPECTED(!strict, "CodecContext is already open.");
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(avcodec_open2(codec_context_, codec_, nullptr) == 0, "Failed to call avcodec_open2.");
  is_open_ = true;
  return Status::OK();
}

Status CodecContext::SendPacketAndRecv(const std::shared_ptr<Packet> &packet,
                                       std::vector<std::shared_ptr<Frame>> *frames) {
  RETURN_UNEXPECTED_IF_NULL(frames);
  CHECK_FAIL_RETURN_UNEXPECTED(avcodec_send_packet(codec_context_, packet->GetAVPacket()) == 0,
                               "Failed to call avcodec_send_packet.");

  while (true) {
    std::shared_ptr<Frame> frame;
    RETURN_IF_NOT_OK(RecvFrame(&frame));
    if (frame != nullptr) {
      (*frames).emplace_back(frame);
    } else {
      break;
    }
  }
  return Status::OK();
}

Status CodecContext::RecvFrame(std::shared_ptr<Frame> *frame) {
  RETURN_UNEXPECTED_IF_NULL(frame);
  if (next_frame_ == nullptr) {
    RETURN_IF_NOT_OK(AllocNextFrame(&next_frame_));
  }

  int res = avcodec_receive_frame(codec_context_, next_frame_->GetAVFrame());
  if (res == AVERROR(EAGAIN) || res == AVERROR_EOF) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(res == 0, "Failed to call avcodec_receive_frame.");

  *frame = next_frame_;
  next_frame_ = nullptr;

  return Status::OK();
}

const char *CodecContext::GetName() { return codec_->name; }

int CodecContext::GetBitRate() {
  if (codec_context_->bit_rate > 0) {
    return codec_context_->bit_rate;
  }
  return -1;
}

int CodecContext::GetFlags() { return codec_context_->flags; }

std::string CodecContext::GetExtradata() const {
  if (codec_context_ == nullptr) {
    return "";
  } else if (codec_context_->extradata_size > 0) {
    return std::string(reinterpret_cast<char *>(codec_context_->extradata), codec_context_->extradata_size);
  } else {
    return "";
  }
}

void CodecContext::SetStreamIndex(int32_t stream_index) { stream_index_ = stream_index; }

void CodecContext::FlushBuffers() {
  if (is_open_) {
    avcodec_flush_buffers(codec_context_);
  }
}

Status CodecContext::AllocNextFrame(std::shared_ptr<Frame> *frame) {
  RETURN_STATUS_UNEXPECTED("Base CodecContext cannot decode.");
  return Status::OK();
}

Status VideoCodecContext::Init(AVCodecContext *codec_context, const AVCodec *codec) {
  RETURN_IF_NOT_OK(CodecContext::Init(codec_context, codec));
  RETURN_IF_NOT_OK(BuildFormat());
  return Status::OK();
}

Status VideoCodecContext::AllocNextFrame(std::shared_ptr<Frame> *frame) { return AllocVideoFrame(frame); }

Status VideoCodecContext::BuildFormat() {
  RETURN_IF_NOT_OK(
    GetVideoFormat(codec_context_->pix_fmt, codec_context_->width, codec_context_->height, &video_format_));
  return Status::OK();
}

int VideoCodecContext::GetWidth() const {
  if (codec_context_ == nullptr) {
    return 0;
  }
  return codec_context_->width;
}

int VideoCodecContext::GetHeight() const {
  if (codec_context_ == nullptr) {
    return 0;
  }
  return codec_context_->height;
}

Status AudioCodecContext::AllocNextFrame(std::shared_ptr<Frame> *frame) { return AllocAudioFrame(frame); }

int &AudioCodecContext::GetRate() { return codec_context_->sample_rate; }

Status AudioCodecContext::SetupDecodedFrame(const std::shared_ptr<Frame> &frame,
                                            const std::shared_ptr<Packet> &packet) {
  RETURN_IF_NOT_OK(CodecContext::SetupDecodedFrame(frame, packet));
  RETURN_IF_NOT_OK(frame->InitUserAttributes());
  return Status::OK();
}

Status WrapCodecContext(AVCodecContext *codec_context, const AVCodec *codec,
                        std::shared_ptr<CodecContext> *out_codec_context) {
  RETURN_UNEXPECTED_IF_NULL(out_codec_context);
  switch (codec_context->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
      *out_codec_context = std::make_shared<VideoCodecContext>();
      break;
    case AVMEDIA_TYPE_AUDIO:
      *out_codec_context = std::make_shared<AudioCodecContext>();
      break;
    default:
      *out_codec_context = std::make_shared<CodecContext>();
  }
  RETURN_IF_NOT_OK((*out_codec_context)->Init(codec_context, codec));
  return Status::OK();
}
}  // namespace mindspore::dataset
