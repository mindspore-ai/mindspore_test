/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/tensordump_utils.h"
#include <string>
#include <fstream>
#include "debug/data_dump/npy_header.h"
#include "ir/tensor.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::device::ascend {
namespace {

void SaveTensor2NPY(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr) {
  if (tensor_ptr->data_type_c() == TypeId::kNumberTypeBFloat16) {
    auto bfloat16_tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(*tensor_ptr, TypeId::kNumberTypeFloat32);
    tensor_ptr = bfloat16_tensor_ptr;
  }
  std::string npy_header = GenerateNpyHeader(tensor_ptr->shape(), tensor_ptr->data_type());
  if (!npy_header.empty()) {
    ChangeFileMode(file_name, S_IWUSR);
    std::fstream output{file_name, std::ios::out | std::ios::trunc | std::ios::binary};
    if (!output.is_open()) {
      MS_LOG(ERROR) << "For 'TensorDump' ops, open " << file_name << " file failed, the args of 'file' is invalid.";
      return;
    }
    output << npy_header;
    (void)output.write(reinterpret_cast<const char *>(tensor_ptr->data_c()), SizeToLong(tensor_ptr->Size()));
    if (output.bad()) {
      output.close();
      MS_LOG(ERROR) << "For 'TensorDump' ops, write mem to " << file_name << " failed.";
      return;
    }
    output.close();
    ChangeFileMode(file_name, S_IRUSR);
  } else {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the type of " << TypeIdToType(tensor_ptr->data_type())->ToString()
                  << " not support dump.";
  }
}

bool EndsWith(const std::string &s, const std::string &sub) {
  if (s.length() < sub.length()) {
    return false;
  }
  return s.rfind(sub) == (s.length() - sub.length()) ? true : false;
}

}  // namespace

AsyncFileWriter::AsyncFileWriter(size_t thread_nums) { threads.reserve(thread_nums); }

AsyncFileWriter::~AsyncFileWriter() {
  stop.store(true, std::memory_order_acq_rel);
  cv.notify_all();
  for (auto &thread : threads) {
    if (thread.joinable()) {
      MS_LOG(INFO) << "TensorDump join file writer threads";
      thread.join();
    }
  }
}

void AsyncFileWriter::Submit(std::function<void()> func) {
  if (!threads_started.exchange(true)) {
    MS_LOG(INFO) << "Create AsyncFileWriter threads.";
    for (size_t i = 0; i < threads.capacity(); ++i) {
      threads.emplace_back(&AsyncFileWriter::WorkerThread, this);
    }
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    tasks.push(func);
  }
  cv.notify_one();
}

void AsyncFileWriter::WorkerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] { return stop || !tasks.empty(); });
      if (stop && tasks.empty()) {
        return;
      }
      task = tasks.front();
      tasks.pop();
    }
    task();
  }
}

std::string TensorDumpUtils::TensorNameToArrayName(const std::string &tensor_path, const std::string &data_type) {
  static size_t name_id = 0;
  std::string npy_suffix{".npy"};
  std::string separator{"_"};
  std::optional<std::string> parent_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(tensor_path, &parent_path, &file_name);
  if (!parent_path.has_value()) {
    parent_path = ".";
  }
  std::optional<std::string> realpath = FileUtils::CreateNotExistDirs(parent_path.value());
  if (EndsWith(file_name.value(), npy_suffix)) {
    file_name = file_name.value().substr(0, file_name.value().length() - npy_suffix.length());
  }
  std::optional<std::string> new_file_name =
    file_name.value() + separator + data_type + separator + std::to_string(name_id++);
  new_file_name.value() += npy_suffix;
  std::optional<std::string> new_file_path;
  FileUtils::ConcatDirAndFileName(&realpath, &new_file_name, &new_file_path);
  MS_LOG(INFO) << "For 'TensorDump' ops, dump file path is " << new_file_path.value();
  return new_file_path.value();
}

TensorDumpUtils &TensorDumpUtils::GetInstance() {
  static TensorDumpUtils instance;
  return instance;
}

void TensorDumpUtils::AsyncSaveDatasetToNpyFile(const ScopeAclTdtDataset &dataset) {
  std::string tensor_name = dataset.GetDatasetName();
  MS_LOG(INFO) << "For 'TensorDump' ops, acltdt received Tensor name is " << tensor_name;
  if (tensor_name.empty()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the args of 'file' is empty, skip this data.";
    return;
  }

  for (auto data_elem : dataset.GetDataItems()) {
    if (std::holds_alternative<std::string>(data_elem)) {
      MS_LOG(WARNING) << "Ignore data of string type: " << std::get<std::string>(data_elem);
    }
    auto tensor_ptr = std::get<mindspore::tensor::TensorPtr>(data_elem);
    std::string data_type = TypeIdToType(tensor_ptr->data_type())->ToString();
    std::transform(data_type.begin(), data_type.end(), data_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    auto file_name = TensorNameToArrayName(tensor_name, data_type);
    file_writer.Submit(std::bind(SaveTensor2NPY, file_name, tensor_ptr));
  }
}

}  // namespace mindspore::device::ascend
