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

#include "minddata/dataset/vision/kernels/pyav/packet.h"

#include "minddata/dataset/vision/kernels/pyav/container.h"
#include "minddata/dataset/vision/kernels/pyav/frame.h"
#include "minddata/dataset/vision/kernels/pyav/stream.h"

namespace mindspore::dataset {
Packet::Packet() : packet_(av_packet_alloc()), stream_(nullptr), time_base_(nullptr) {}

Packet::~Packet() { av_packet_free(&packet_); }

Status Packet::Decode(std::vector<std::shared_ptr<Frame>> *frames) {
  RETURN_UNEXPECTED_IF_NULL(stream_);
  return stream_->Decode(shared_from_this(), frames);
}

int Packet::GetPTS() const {
  if (packet_->pts != AV_NOPTS_VALUE) {
    return packet_->pts;
  }
  return -1;
}

int Packet::GetDTS() const {
  if (packet_->dts != AV_NOPTS_VALUE) {
    return packet_->dts;
  }
  return -1;
}

bool Packet::IsKeyFrame() const { return static_cast<bool>(static_cast<uint32_t>(packet_->flags) & AV_PKT_FLAG_KEY); }

AVRational *Packet::GetTimeBase() { return time_base_; }

AVPacket *Packet::GetAVPacket() { return packet_; }
}  // namespace mindspore::dataset
