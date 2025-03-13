/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/backend/debug/data_dump/dump_utils.h"
#include <dirent.h>
#ifdef ENABLE_DEBUGGER
#include <sys/stat.h>
#endif
#include <map>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>

#include "include/common/utils/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#include "include/backend/debug/tensor_data.h"
#endif
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "utils/file_utils.h"

using mindspore::runtime::DeviceTensorStore;

namespace mindspore {
static std::vector<std::string> g_overflow_operators;

bool SplitInt8ToInt4x2(const void *int4_data, size_t in_data_len, void *int8_data, size_t out_data_len) {
  if (in_data_len * 2 != out_data_len) {
    MS_LOG(ERROR) << "The input data length and output data length is not match, input data length: " << in_data_len
                  << ", output data length: " << out_data_len
                  << ". If sample_mode is set to 1, then sample_num must set to Integer multiples of 2 to save tensor "
                     "with int4 data type.";
    return false;
  }
  int8_t *src_data = static_cast<int8_t *>(const_cast<void *>(int4_data));
  int8_t *dst_data = static_cast<int8_t *>(int8_data);
  for (size_t i = 0; i < in_data_len; ++i) {
    int8_t s = *src_data;
    int8_t t = s & 0xf;
    // keep the sign bit not change
    int8_t sign_bit = (t & 0x08) >> 3;
    if (sign_bit == 1) {
      t = t | 0xf0;
    } else if (sign_bit == 0) {
      t = t & 0x0f;
    } else {
      MS_LOG(ERROR) << "Error occur.";
      return false;
    }
    if (t < -8 || t > 7) {
      MS_LOG(ERROR) << "Error occurred when convert int4 to int8 data.";
      return false;
    }
    *dst_data = t;
    ++dst_data;
    t = s >> 4;
    sign_bit = (t & 0x08) >> 3;
    if (sign_bit == 1) {
      t = t | 0xf0;
    } else if (sign_bit == 0) {
      t = t & 0x0f;
    } else {
      MS_LOG(ERROR) << "Error occur.";
      return false;
    }
    if (t < -8 || t > 7) {
      MS_LOG(ERROR) << "Error occurred when convert int4 to int8 data.";
      return false;
    }
    *dst_data = t;
    ++dst_data;
    ++src_data;
  }
  return true;
}

void SplitUint1x8ToUint8s(const void *in_data, size_t in_data_len, ShapeVector shape, void *out_data) {
  MS_EXCEPTION_IF_NULL(in_data);
  MS_EXCEPTION_IF_NULL(out_data);
  MS_EXCEPTION_IF_ZERO("in_data_len", in_data_len);
  auto element_num = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  const int64_t elemnum_per_byte = 8;
  auto elemnum_last_byte = element_num % elemnum_per_byte;

  const uint8_t *src_data = static_cast<const uint8_t *>(in_data);
  uint8_t *dst_data = static_cast<uint8_t *>(out_data);
  for (size_t i = 0; i < in_data_len - 1; ++i) {
    for (int j = 7; j >= 0; --j) {
      *dst_data = (*src_data >> j) & 1;
      ++dst_data;
    }
    ++src_data;
  }

  // Handles cases where the number of in_data elements is not a multiple of 8
  for (int j = 7; j >= elemnum_last_byte; --j) {
    *dst_data = (*src_data >> j) & 1;
    ++dst_data;
  }
}

std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id, bool is_cst) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  if (is_cst) {
    dump_path += ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/constants/");
  } else {
    dump_path +=
      ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/" + iterator + "/");
  }
  return dump_path;
}

void GetFileKernelName(NotNull<std::string *> kernel_name) {
  const std::string strsrc_to_replace[4] = {"/", "\\", ".", " "};
  const std::string strdst = "_";

  for (const auto &strsrc : strsrc_to_replace) {
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();
    while ((pos = kernel_name->find(strsrc, pos)) != std::string::npos) {
      kernel_name->replace(pos, srclen, strdst);
      pos += dstlen;
    }
  }
}

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> const int_shapes, bool trans_flag) {
  if (trans_flag) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<None>(node)) {
      return;
    }
    *int_shapes = AnfAlgo::GetRuntimePaddingShape(node, index);
  } else {
    *int_shapes = AnfAlgo::GetOutputDeviceShape(node, index);
  }
}

const DeviceTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                       NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type) {
  const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(node.get());
  if (device_tensors.size() < 1) {
    return nullptr;
  }
  auto device_addr = device_tensors[0];
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool trans_flag = dump_json_parser.trans_flag();
  auto ref_node = device_addr->GetNodeIndex().first;
  MS_EXCEPTION_IF_NULL(ref_node);
  GetDumpIntShape(ref_node, kParameterOutputIndex, int_shapes, trans_flag);
  *host_type = common::AnfAlgo::GetOutputInferDataType(ref_node, kParameterOutputIndex);
  *device_type = AnfAlgo::GetOutputDeviceDataType(ref_node, kParameterOutputIndex);
  return device_addr;
}

bool CPUDumpMemToFile(const device::DeviceAddress &addr, const std::string &filepath, const std::string &,
                      const ShapeVector &host_shape, TypeId host_type, bool) {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string path = filepath + '.' + addr.format() + "." + TypeIdToString(host_type);
  MS_LOG(DEBUG) << "E2E Dump path is " << path;
  if (addr.GetSize() == 0) {
    MS_LOG(INFO) << "Data size is 0 for file: " << path << ", no need to dump.";
    return true;
  }
  if (addr.GetPtr() == nullptr) {
    MS_LOG(WARNING) << "Data is nullptr for file: " << path << ", skip it.";
    return true;
  }
  ret = DumpJsonParser::DumpToFile(path, addr.GetPtr(), addr.GetSize(), host_shape, host_type);
  if (!ret) {
    MS_LOG(ERROR) << "Dump to file failed";
  }
  return ret;
}

bool AscendDumpMemToFile(const device::DeviceAddress &addr, const std::string &filepath, const std::string &host_fmt,
                         const ShapeVector &host_shape, TypeId host_type, bool trans_flag) {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  if (addr.GetSize() == 0) {
    MS_LOG(INFO) << "the operator in filepath: " << filepath << ", size == 0";
    return true;
  }
  if (trans_flag) {
    std::string path = filepath + '.' + host_fmt;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
        host_type == kNumberTypeComplex64) {
      MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
      return false;
    }
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    MS_EXCEPTION_IF_NULL(out_tensor);
    size_t host_size = LongToSize(out_tensor->data().nbytes());
    ret = addr.SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size, host_shape, host_type);
  } else {
    auto host_tmp = std::vector<uint8_t>(addr.GetSize());
    addr.SyncDeviceToHost(addr.GetSize(), host_tmp.data());
    std::string path = filepath + '.' + addr.format();
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), addr.GetSize(), host_shape, addr.type_id());
  }
  return ret;
}

void DumpMemToFile(const std::string &file_path, const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                   const TypeId &type, bool trans_flag) {
  auto format = kOpFormat_DEFAULT;
  bool ret = false;
  if (addr.GetDeviceType() == device::DeviceType::kCPU) {
    ret = CPUDumpMemToFile(addr, file_path, format, int_shapes, type, trans_flag);
  } else if (addr.GetDeviceType() == device::DeviceType::kAscend) {
    ret = AscendDumpMemToFile(addr, file_path, format, int_shapes, type, trans_flag);
  }
  if (!ret) {
    MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << file_path << ", host_format:" << format
                  << ".!";
  }
}

void DumpToFile(const std::string &file_name, const std::string &dump_str) {
  if (dump_str.empty()) {
    MS_LOG(ERROR) << "Failed to dump empty tensor data.";
    return;
  }

  auto real_path = Common::CreatePrefixPath(file_name);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "CreatePrefixPath failed.";
    return;
  }
  std::string real_path_str = real_path.value();
  ChangeFileMode(real_path_str, S_IWUSR);
  std::ofstream file(real_path_str, std::ofstream::out | std::ofstream::trunc);
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "Open file " << real_path_str << "failed: " << ErrnoToString(errno);
  }
  file << dump_str;
  if (file.bad()) {
    file.close();
    MS_LOG(EXCEPTION) << "Dump string to file " << real_path_str << " failed: " << ErrnoToString(errno);
  }
  file.close();
  ChangeFileMode(real_path_str, S_IRUSR);
}
#ifdef ENABLE_DEBUGGER
bool LoadMemToHost(const device::DeviceAddress &addr, const std::string &tensor_name, int execution_order,
                   const std::string &host_fmt, const ShapeVector &host_shape, TypeId host_type, size_t slot,
                   bool keep_prev, uint32_t root_graph_id, bool force_update, bool trans_flag, bool async_copy) {
  bool ret = false;
  if (addr.GetSize() == 0) {
    MS_LOG(INFO) << tensor_name << " size is 0, skip it.";
    return true;
  }
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->TensorExistsInCurrent(tensor_name) && !force_update) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin || host_type == kNumberTypeComplex64) {
    MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
    return false;
  }
  auto out_tensor = addr.LoadMemToHost(tensor_name, host_shape, host_type, trans_flag, async_copy);
  if (!out_tensor) {
    MS_LOG(ERROR) << tensor_name << " load mem to host failed.";
    return false;
  }
  if (!out_tensor->DataSize()) {
    MS_LOG(INFO) << tensor_name << " datasize is 0, skip it.";
    return true;
  }
  std::string tensor_format = trans_flag ? host_fmt : addr.format();
  size_t host_size = LongToSize(out_tensor->data().nbytes());
  if (host_type == kNumberTypeInt4) {
    const int int4_nums_per_byte = 2;
    host_size = out_tensor->DataSize() / int4_nums_per_byte;
  }
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(host_size);
  tensor_data->SetType(host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetRootGraphId(root_graph_id);
  tensor_data->SetFormat(tensor_format);
  ret = debugger->LoadNewTensor(tensor_data, keep_prev);
  MS_LOG(INFO) << "Load tensor '" << tensor_name << "' into debugger tensor loader successfully: format("
               << tensor_format << ").";
  return ret;
}
#endif
}  // namespace mindspore
