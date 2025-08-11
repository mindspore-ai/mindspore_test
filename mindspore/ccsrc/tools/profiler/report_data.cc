/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/profiler/report_data.h"

namespace mindspore {
namespace profiler {
namespace ascend {

std::vector<uint8_t> OpRangeData::encode() {
  std::vector<uint8_t> tlvBytes;

  // Fixed data
  // 5 * 8 = 40 bytes
  encodeFixedData<uint64_t>(thread_id, tlvBytes);
  encodeFixedData<uint64_t>(flow_id, tlvBytes);
  encodeFixedData<uint64_t>(step, tlvBytes);
  encodeFixedData<uint64_t>(start_time_ns, tlvBytes);
  encodeFixedData<uint64_t>(end_time_ns, tlvBytes);
  // 4 bytes
  encodeFixedData<int32_t>(process_id, tlvBytes);
  // 2 * 2 = 4 bytes
  encodeFixedData<uint16_t>(module_index, tlvBytes);
  encodeFixedData<uint16_t>(event_index, tlvBytes);
  encodeFixedData<uint16_t>(stage_index, tlvBytes);
  // 1 byte
  encodeFixedData<int8_t>(level, tlvBytes);
  // 2 bytes
  encodeFixedData<bool>(is_graph_data, tlvBytes);
  encodeFixedData<bool>(is_stage, tlvBytes);
  encodeFixedData<bool>(is_stack, tlvBytes);
  // Dynamic length data
  encodeStrData(static_cast<uint16_t>(OpRangeDataType::NAME), op_name, tlvBytes);
  encodeStrData(static_cast<uint16_t>(OpRangeDataType::FULL_NAME), op_full_name, tlvBytes);
  if (!custom_info.empty()) {
    encodeStrMapData(static_cast<uint16_t>(OpRangeDataType::CUSTOM_INFO), custom_info, tlvBytes);
  }
  encodeStrData(static_cast<uint16_t>(OpRangeDataType::MODULE_GRAPH), module_graph, tlvBytes);
  encodeStrData(static_cast<uint16_t>(OpRangeDataType::EVENT_GRAPH), event_graph, tlvBytes);

  std::vector<uint8_t> resultTLV;
  size_t totalSize = sizeof(uint16_t) + sizeof(uint32_t) + tlvBytes.size();
  resultTLV.reserve(totalSize);

  uint16_t dataType = static_cast<uint16_t>(ReportFileType::OP_RANGE);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    resultTLV.push_back((dataType >> (i * kBitsPerByte)) & 0xff);
  }
  uint32_t length = tlvBytes.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    resultTLV.push_back((length >> (i * kBitsPerByte)) & 0xff);
  }
  resultTLV.insert(resultTLV.end(), tlvBytes.cbegin(), tlvBytes.cend());
  return resultTLV;
}

std::vector<uint8_t> RecordShapesData::encode() {
  std::vector<uint8_t> tlvBytes;

  // Fixed data
  // Dynamic length data
  encodeStrData(static_cast<uint16_t>(RecordShapesDataType::NAME), op_name, tlvBytes);
  encodeStrData(static_cast<uint16_t>(RecordShapesDataType::INPUT_SHAPES), input_shapes, tlvBytes);
  encodeStrData(static_cast<uint16_t>(RecordShapesDataType::INPUT_TYPE), input_type, tlvBytes);

  std::vector<uint8_t> resultTLV;
  size_t totalSize = sizeof(uint16_t) + sizeof(uint32_t) + tlvBytes.size();
  resultTLV.reserve(totalSize);

  uint16_t dataType = static_cast<uint16_t>(ReportFileType::RECORD_SHAPES);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    resultTLV.push_back((dataType >> (i * kBitsPerByte)) & 0xff);
  }
  uint32_t length = tlvBytes.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    resultTLV.push_back((length >> (i * kBitsPerByte)) & 0xff);
  }
  resultTLV.insert(resultTLV.end(), tlvBytes.cbegin(), tlvBytes.cend());
  return resultTLV;
}

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
