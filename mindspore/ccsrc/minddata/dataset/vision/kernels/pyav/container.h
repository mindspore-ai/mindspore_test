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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTAINER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTAINER_H_

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class Frame;
class Packet;
class Stream;
class VideoStream;
class AudioStream;

class StreamContainer {
 public:
  StreamContainer() = default;

  Status AddStream(std::shared_ptr<Stream> stream);

  const std::vector<std::shared_ptr<VideoStream>> &GetVideo() { return video_; }

  const std::vector<std::shared_ptr<AudioStream>> &GetAudio() { return audio_; }

  size_t Size() { return streams_.size(); }

  std::shared_ptr<Stream> Get(int streams = -1, int video = -1, int audio = -1) const;

  std::shared_ptr<Stream> operator[](size_t index);

 private:
  std::vector<std::shared_ptr<Stream>> streams_;
  std::vector<std::shared_ptr<VideoStream>> video_;
  std::vector<std::shared_ptr<AudioStream>> audio_;
  std::vector<std::shared_ptr<Stream>> other_;
};

class Container : public std::enable_shared_from_this<Container> {
 public:
  explicit Container(const std::string &file);

  Container(const Container &other) = delete;

  Container(Container &&other) = delete;

  Container &operator=(const Container &other) = delete;

  Container &operator=(Container &&other) = delete;

  ~Container();

  Status Init();

  void Close();

  Status Seek(int64_t offset, bool backward, bool any_frame, const std::shared_ptr<Stream> &stream);

  Status Demux(const std::shared_ptr<Stream> &stream, std::vector<std::shared_ptr<Packet>> *packets);

  Status Decode(const std::shared_ptr<Stream> &stream, std::vector<std::shared_ptr<Frame>> *frames);

  const StreamContainer &GetStreams() const { return streams_; }

  const AVFormatContext *GetFormatContext() const { return format_context_; }

 private:
  Status AssertOpen() const;

  Status FlushBuffers();

  std::string name_;
  bool input_was_opened_;
  AVFormatContext *format_context_;
  StreamContainer streams_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_CONTAINER_H_
