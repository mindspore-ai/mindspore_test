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
#include <memory>

#include "debug/data_dump/npy_header.h"
#include "ir/tensor.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "include/backend/debug/data_dump/tensordump_control.h"
#include "backend/common/session/session_basic.h"

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

}  // namespace

TensorDumpUtils &TensorDumpUtils::GetInstance() {
  static TensorDumpUtils instance;
  return instance;
}

void TensorDumpUtils::SaveDatasetToNpyFile(const ScopeAclTdtDataset &dataset) {
  std::string tensor_name = dataset.GetDatasetName();
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, acltdt received Tensor name is " << tensor_name;
  if (tensor_name.empty()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the args of 'file' is empty, skip this data.";
    return;
  }
  auto data_items = dataset.GetDataItems();
  if (data_items.size() != 1) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the args of 'input_x' only support one input, bug got "
                  << data_items.size();
    return;
  }
  auto data_elem = data_items.front();
  if (std::holds_alternative<std::string>(data_elem)) {
    MS_LOG(WARNING) << "Ignore data of string type: " << std::get<std::string>(data_elem);
  }
  auto tensor_ptr = std::get<mindspore::tensor::TensorPtr>(data_elem);
  std::string data_type = TypeIdToType(tensor_ptr->data_type())->ToString();
  auto file_name = TensorDumpStepManager::GetInstance().ProcessFileName(tensor_name, data_type);
  if (file_name.empty()) {  // update step or not need dump
    return;
  }
  SaveTensor2NPY(file_name, tensor_ptr);
}

}  // namespace mindspore::device::ascend
