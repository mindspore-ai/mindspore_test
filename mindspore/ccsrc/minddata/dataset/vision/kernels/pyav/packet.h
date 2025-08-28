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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_PACKET_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_PACKET_H_

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <memory>
#include <vector>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class Frame;
class Stream;

class Packet : public std::enable_shared_from_this<Packet> {
 public:
  friend class Container;

  Packet();

  ~Packet();

  Status Decode(std::vector<std::shared_ptr<Frame>> *frames);

  int GetPTS() const;

  int GetDTS() const;

  AVRational *GetTimeBase();

  AVPacket *GetAVPacket();

  bool IsKeyFrame() const;

 private:
  AVPacket *packet_;
  std::shared_ptr<Stream> stream_;
  AVRational *time_base_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PYAV_PACKET_H_
