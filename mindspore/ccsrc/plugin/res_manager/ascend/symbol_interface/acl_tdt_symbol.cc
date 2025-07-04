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
#include "acl_tdt_symbol.h"
#include <string>
#include <vector>
#include "symbol_utils.h"

namespace mindspore::device::ascend {

acltdtAddDataItemFunObj acltdtAddDataItem_ = nullptr;
acltdtCleanChannelFunObj acltdtCleanChannel_ = nullptr;
acltdtCreateChannelFunObj acltdtCreateChannel_ = nullptr;
acltdtCreateChannelWithCapacityFunObj acltdtCreateChannelWithCapacity_ = nullptr;
acltdtCreateDataItemFunObj acltdtCreateDataItem_ = nullptr;
acltdtCreateDatasetFunObj acltdtCreateDataset_ = nullptr;
acltdtDestroyChannelFunObj acltdtDestroyChannel_ = nullptr;
acltdtDestroyDataItemFunObj acltdtDestroyDataItem_ = nullptr;
acltdtDestroyDatasetFunObj acltdtDestroyDataset_ = nullptr;
acltdtGetDataAddrFromItemFunObj acltdtGetDataAddrFromItem_ = nullptr;
acltdtGetDataItemFunObj acltdtGetDataItem_ = nullptr;
acltdtGetDatasetNameFunObj acltdtGetDatasetName_ = nullptr;
acltdtGetDatasetSizeFunObj acltdtGetDatasetSize_ = nullptr;
acltdtGetDataSizeFromItemFunObj acltdtGetDataSizeFromItem_ = nullptr;
acltdtGetDataTypeFromItemFunObj acltdtGetDataTypeFromItem_ = nullptr;
acltdtGetDimNumFromItemFunObj acltdtGetDimNumFromItem_ = nullptr;
acltdtGetDimsFromItemFunObj acltdtGetDimsFromItem_ = nullptr;
acltdtGetTensorTypeFromItemFunObj acltdtGetTensorTypeFromItem_ = nullptr;
acltdtGetSliceInfoFromItemFunObj acltdtGetSliceInfoFromItem_ = nullptr;
acltdtQueryChannelSizeFunObj acltdtQueryChannelSize_ = nullptr;
acltdtReceiveTensorFunObj acltdtReceiveTensor_ = nullptr;
acltdtSendTensorFunObj acltdtSendTensor_ = nullptr;
acltdtStopChannelFunObj acltdtStopChannel_ = nullptr;

void LoadAcltdtApiSymbol(const std::string &ascend_path) {
  const std::vector<std::string> depend_libs = {"libacl_tdt_queue.so"};
  for (const auto &dep_lib : depend_libs) {
    (void)GetLibHandler(ascend_path + "lib64/" + dep_lib);
  }

  std::string aclrt_tdt_path = ascend_path + "lib64/libacl_tdt_channel.so";
  auto handler = GetLibHandler(aclrt_tdt_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << aclrt_tdt_path << " failed!" << dlerror();
    return;
  }
  acltdtAddDataItem_ = DlsymAscendFuncObj(acltdtAddDataItem, handler);
  acltdtCleanChannel_ = DlsymAscendFuncObj(acltdtCleanChannel, handler);
  acltdtCreateChannel_ = DlsymAscendFuncObj(acltdtCreateChannel, handler);
  acltdtCreateChannelWithCapacity_ = DlsymAscendFuncObj(acltdtCreateChannelWithCapacity, handler);
  acltdtCreateDataItem_ = DlsymAscendFuncObj(acltdtCreateDataItem, handler);
  acltdtCreateDataset_ = DlsymAscendFuncObj(acltdtCreateDataset, handler);
  acltdtDestroyChannel_ = DlsymAscendFuncObj(acltdtDestroyChannel, handler);
  acltdtDestroyDataItem_ = DlsymAscendFuncObj(acltdtDestroyDataItem, handler);
  acltdtDestroyDataset_ = DlsymAscendFuncObj(acltdtDestroyDataset, handler);
  acltdtGetDataAddrFromItem_ = DlsymAscendFuncObj(acltdtGetDataAddrFromItem, handler);
  acltdtGetDataItem_ = DlsymAscendFuncObj(acltdtGetDataItem, handler);
  acltdtGetDatasetName_ = DlsymAscendFuncObj(acltdtGetDatasetName, handler);
  acltdtGetDatasetSize_ = DlsymAscendFuncObj(acltdtGetDatasetSize, handler);
  acltdtGetDataSizeFromItem_ = DlsymAscendFuncObj(acltdtGetDataSizeFromItem, handler);
  acltdtGetDataTypeFromItem_ = DlsymAscendFuncObj(acltdtGetDataTypeFromItem, handler);
  acltdtGetDimNumFromItem_ = DlsymAscendFuncObj(acltdtGetDimNumFromItem, handler);
  acltdtGetDimsFromItem_ = DlsymAscendFuncObj(acltdtGetDimsFromItem, handler);
  acltdtGetTensorTypeFromItem_ = DlsymAscendFuncObj(acltdtGetTensorTypeFromItem, handler);
  acltdtGetSliceInfoFromItem_ = DlsymAscendFuncObj(acltdtGetSliceInfoFromItem, handler);
  acltdtQueryChannelSize_ = DlsymAscendFuncObj(acltdtQueryChannelSize, handler);
  acltdtReceiveTensor_ = DlsymAscendFuncObj(acltdtReceiveTensor, handler);
  acltdtSendTensor_ = DlsymAscendFuncObj(acltdtSendTensor, handler);
  acltdtStopChannel_ = DlsymAscendFuncObj(acltdtStopChannel, handler);
  MS_LOG(INFO) << "Load acl tdt api success!";
}

void LoadSpecialSimulationTdtApi() {
  acltdtQueryChannelSize_ = [](const acltdtChannelHandle *handle, size_t *ret_size_ptr) {
    if (handle == nullptr) {
      MS_LOG(INFO) << "Empty handle!";
    }
    if (ret_size_ptr != nullptr) {
      *ret_size_ptr = 1;
    }
    return ACL_SUCCESS;
  };
}

void LoadSimulationTdtApi() {
  ASSIGN_SIMU(acltdtAddDataItem);
  ASSIGN_SIMU(acltdtCreateChannel);
  ASSIGN_SIMU(acltdtCreateChannelWithCapacity);
  ASSIGN_SIMU(acltdtCreateDataItem);
  ASSIGN_SIMU(acltdtCreateDataset);
  ASSIGN_SIMU(acltdtDestroyChannel);
  ASSIGN_SIMU(acltdtDestroyDataItem);
  ASSIGN_SIMU(acltdtDestroyDataset);
  ASSIGN_SIMU(acltdtGetDataAddrFromItem);
  ASSIGN_SIMU(acltdtGetDataItem);
  ASSIGN_SIMU(acltdtGetDatasetName);
  ASSIGN_SIMU(acltdtGetDatasetSize);
  ASSIGN_SIMU(acltdtGetDataSizeFromItem);
  ASSIGN_SIMU(acltdtGetDataTypeFromItem);
  ASSIGN_SIMU(acltdtGetDimNumFromItem);
  ASSIGN_SIMU(acltdtGetDimsFromItem);
  ASSIGN_SIMU(acltdtGetTensorTypeFromItem);
  ASSIGN_SIMU(acltdtGetSliceInfoFromItem);
  ASSIGN_SIMU(acltdtQueryChannelSize);
  ASSIGN_SIMU(acltdtReceiveTensor);
  ASSIGN_SIMU(acltdtSendTensor);
  ASSIGN_SIMU(acltdtStopChannel);
  LoadSpecialSimulationTdtApi();
}
}  // namespace mindspore::device::ascend
