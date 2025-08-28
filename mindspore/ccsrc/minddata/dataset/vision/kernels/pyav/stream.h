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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_STREAM_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_STREAM_H_

extern "C" {
#include <libavformat/avformat.h>
}

#include <memory>
#include <vector>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class CodecContext;
class Container;
class Frame;
class Packet;

class Stream {
 public:
  Stream() = default;

  virtual void Init(std::shared_ptr<Container> container, AVStream *av_stream,
                    std::shared_ptr<CodecContext> codec_context);

  virtual Status Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames);

  virtual AVRational *GetTimeBase();

  virtual int GetStartTime();

  virtual int GetBitRate() const;

  virtual int GetFlags() const;

  virtual int GetFrames() const;

  virtual int GetDuration() const;

  virtual const AVStream *GetAVStream() const { return stream_; }

  virtual int GetIndex() const { return stream_->index; }

  virtual const std::shared_ptr<CodecContext> &GetCodecContext() const;

  virtual const char *GetType() const;

 protected:
  std::shared_ptr<Container> container_;
  AVStream *stream_;
  std::shared_ptr<CodecContext> codec_context_;
};

class VideoStream : public Stream {
 public:
  Status Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) override;

  AVRational *GetAverageRate() const;

  const char *GetName() const;

  int GetWidth() const;

  int GetHeight() const;
};

class AudioStream : public Stream {
 public:
  Status Decode(const std::shared_ptr<Packet> &packet, std::vector<std::shared_ptr<Frame>> *frames) override;

  int GetRate();
};

Status WrapStream(std::shared_ptr<Container> av_container, AVStream *av_stream,
                  std::shared_ptr<CodecContext> codec_context, std::shared_ptr<Stream> *out_stream);
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_STREAM_H_
