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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTEXT_H_

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class Frame;
class Packet;
class VideoFormat;

class CodecContext {
 public:
  CodecContext();

  virtual Status Init(AVCodecContext *codec_context, const AVCodec *codec);

  virtual Status Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames);

  virtual Status Open(bool strict);

  virtual const char *GetName();

  virtual std::string GetExtradata() const;

  virtual void SetStreamIndex(int32_t stream_index);

  virtual void FlushBuffers();

  virtual Status AllocNextFrame(std::shared_ptr<Frame> *frame);

  virtual int GetBitRate();

  virtual int GetFlags();

 protected:
  virtual Status SendPacketAndRecv(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames);

  virtual Status RecvFrame(std::shared_ptr<Frame> *frame);

  virtual Status SetupDecodedFrame(const std::shared_ptr<Frame> &frame, const std::shared_ptr<Packet> &packet);

  int32_t stream_index_;
  AVCodecContext *codec_context_;
  const AVCodec *codec_;
  bool is_open_;
  std::shared_ptr<Frame> next_frame_;
};

class VideoCodecContext : public CodecContext {
 public:
  Status Init(AVCodecContext *codec_context, const AVCodec *codec) override;

  Status AllocNextFrame(std::shared_ptr<Frame> *frame) override;

  int GetWidth() const;

  int GetHeight() const;

 private:
  Status BuildFormat();

  std::shared_ptr<VideoFormat> video_format_;
};

class AudioCodecContext : public CodecContext {
 public:
  Status AllocNextFrame(std::shared_ptr<Frame> *frame) override;

  int &GetRate();

 private:
  Status SetupDecodedFrame(const std::shared_ptr<Frame> &frame, const std::shared_ptr<Packet> &packet) override;
};

Status WrapCodecContext(AVCodecContext *codec_context, const AVCodec *codec,
                        std::shared_ptr<CodecContext> *out_codec_context);
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTEXT_H_
