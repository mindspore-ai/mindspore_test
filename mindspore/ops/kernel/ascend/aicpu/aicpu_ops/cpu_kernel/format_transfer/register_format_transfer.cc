/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
 *
 * https://gitee.com/mindspore/graphengine/blob/v0.1.0-alpha/src/ge/common/formats/format_transfers/format_transfer.cc
 *
 * Additional modifications made by Huawei Technologies Co., Ltd in 2022.
 */
#include "cpu_kernel/format_transfer/register_format_transfer.h"

#include <map>
#include <utility>

namespace aicpu {
namespace formats {
namespace {
struct FormatTransferRegistry {
  void RegisterBuilder(Format src, Format dst, FormatTransferBuilder builder) {
    src_dst_builder[src][dst] = move(builder);
  }
  std::map<Format, std::map<Format, FormatTransferBuilder>> src_dst_builder;
};

FormatTransferRegistry &GetFormatTransferRegistry() {
  static FormatTransferRegistry registry;
  return registry;
}
}  // namespace

FormatTransferRegister::FormatTransferRegister(FormatTransferBuilder builder, Format src, Format dst) {
  GetFormatTransferRegistry().RegisterBuilder(src, dst, move(builder));
}

std::shared_ptr<FormatTransfer> BuildFormatTransfer(const TransArgs &args) {
  auto &registry = GetFormatTransferRegistry();
  auto dst_builder = registry.src_dst_builder.find(args.src_format);
  if (dst_builder == registry.src_dst_builder.end()) {
    return nullptr;
  }
  auto builder_iter = dst_builder->second.find(args.dst_format);
  if (builder_iter == dst_builder->second.end()) {
    return nullptr;
  }
  return builder_iter->second();
}

bool FormatTransferExists(const TransArgs &args) {
  auto &registry = GetFormatTransferRegistry();
  auto dst_builder = registry.src_dst_builder.find(args.src_format);
  if (dst_builder == registry.src_dst_builder.end()) {
    return false;
  }
  return dst_builder->second.count(args.dst_format) > 0;
}
}  // namespace formats
}  // namespace aicpu
