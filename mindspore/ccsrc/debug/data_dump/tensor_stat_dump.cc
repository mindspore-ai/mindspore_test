/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "include/backend/debug/data_dump/tensor_stat_dump.h"
#include <map>
#include "debug/debug_services.h"
#include "debug/utils.h"
#include "include/backend/debug/common/csv_writer.h"
#include "include/backend/debug/debugger/debugger.h"
#include "include/common/debug/common.h"
namespace {
constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kCsvFileName = "statistic.csv";
}  // namespace

namespace mindspore {
TensorStatDump::TensorStatDump(const std::string &op_type, const std::string &op_name, uint32_t task_id,
                               uint32_t stream_id, uint64_t timestamp, bool input, size_t slot,
                               size_t tensor_loader_slot)
    : op_type_{op_type},
      op_name_{op_name},
      task_id_{std::to_string(task_id)},
      stream_id_{std::to_string(stream_id)},
      timestamp_{std::to_string(timestamp)},
      slot_{slot},
      tensor_loader_slot_{tensor_loader_slot} {
  if (input) {
    io_ = kInput;
  } else {
    io_ = kOutput;
  }
}

TensorStatDump::TensorStatDump(const std::string &op_type, const std::string &op_name, const std::string &task_id,
                               const std::string &stream_id, const std::string &timestamp, const std::string &io,
                               size_t slot, size_t tensor_loader_slot, const mindspore::TypeId data_type)
    : op_type_{op_type},
      op_name_{op_name},
      task_id_{task_id},
      stream_id_{stream_id},
      timestamp_{timestamp},
      io_{io},
      slot_{slot},
      tensor_loader_slot_{tensor_loader_slot},
      data_type_{data_type} {
  if (io_ != kInput && io_ != kOutput) {
    MS_LOG(EXCEPTION) << "Cannot instantiate TensorStatDump, io needs to be either " << kInput << " or " << kOutput;
  }
}

bool TensorStatDump::OpenStatisticsFile(const std::string &dump_path) {
  std::string filename = dump_path + "/" + kCsvFileName;
  // try to open file
  CsvWriter &csv = CsvWriter::GetInstance();
  const string csv_header = CsvHeaderUtil::GetInstance().GetStatCsvHeader();
  int retry = 2;
  while (retry > 0) {
    if (csv.OpenFile(filename, csv_header)) {
      break;
    }
    retry--;
  }
  if (retry == 0) {
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return false;
  }
  return true;
}

bool TensorStatDump::DumpTensorStatsToFile(const std::string &original_kernel_name, const std::string &dump_path,
                                           const Debugger *debugger) {
  // get tensor data using debugger
  std::string tensor_loader_name = original_kernel_name + ":" + std::to_string(tensor_loader_slot_);
  std::shared_ptr<TensorData> data = debugger->GetTensor(tensor_loader_name);
  if (data == nullptr) {
    MS_VLOG(VL_DUMP) << "Failed to find " << tensor_loader_name << " in tensor loader, skipping current statistics";
    return false;
  }
  return DumpTensorStatsToFile(dump_path, data);
}

bool TensorStatDump::DumpTensorStatsToFile(const std::string &dump_path, const std::shared_ptr<TensorData> data) {
  if (data == nullptr) {
    MS_VLOG(VL_DUMP) << "Tensor data is empty, skipping current statistics";
    return false;
  }
  std::string type = data->GetTypeString();
  if (type.empty()) {
    type = "unsupported(" + std::to_string(data->GetType()) + ")";
    MS_VLOG(VL_DUMP) << "Unsupported tensor data_type " << type << " for tensor " << data->GetName();
  }
  std::string filename = dump_path + "/" + kCsvFileName;
  // try to open file
  CsvWriter csv;
  std::lock_guard<std::mutex> lock(CsvFileMutexManager::GetInstance().GetCsvMutex(filename));

  const auto csv_header = CsvHeaderUtil::GetInstance().GetStatCsvHeader();
  if (!csv.OpenFile(filename, csv_header)) {
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return false;
  }
  DebugServices::TensorStat stat = DebugServices::GetTensorStatistics(data);
  // write tensor statistics to csv file
  std::ostringstream shape;
  shape << "\"(";
  for (size_t i = 0; i < stat.shape.size(); i++) {
    shape << (i > 0 ? "," : "") << stat.shape[i];
  }
  shape << ")\"";

  csv.WriteToCsv(op_type_);
  csv.WriteToCsv(op_name_);
  csv.WriteToCsv(task_id_);
  csv.WriteToCsv(stream_id_);
  csv.WriteToCsv(timestamp_);
  csv.WriteToCsv(io_);
  csv.WriteToCsv(slot_);
  csv.WriteToCsv(stat.data_size);
  if (data_type_ != mindspore::TypeId::kTypeUnknown) {
    if (data->GetType() == DbgDataType::DT_UINT1) {
      csv.WriteToCsv(type);
    } else {
      csv.WriteToCsv(TypeIdToString(data_type_, true));
    }
  } else {
    csv.WriteToCsv(type);
  }
  csv.WriteToCsv(shape.str());
  stat.UpdateHeaderItemMap();
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  auto statistic_category = dump_json_parser.statistic_category();
  // first several item write to file without endline;
  for (auto &header : statistic_category) {
    auto &item = stat.header_item_map[header];
    csv.WriteToCsv(item);
    MS_VLOG(VL_DUMP) << "Write the :" << header << " into file, value is: " << item;
  }
  csv.WriteToCsv("", true);
  csv.CloseFile();
  return true;
}
}  // namespace mindspore
