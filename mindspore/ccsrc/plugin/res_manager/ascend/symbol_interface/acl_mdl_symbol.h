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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_MDL_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_MDL_SYMBOL_H_
#include <string>
#include "acl/acl_mdl.h"
#include "utils/dlopen_macro.h"

namespace mindspore::device::ascend {
ORIGIN_METHOD(aclmdlAddDatasetBuffer, aclError, aclmdlDataset *, aclDataBuffer *)
ORIGIN_METHOD(aclmdlCreateDataset, aclmdlDataset *);
ORIGIN_METHOD(aclmdlCreateDesc, aclmdlDesc *)
ORIGIN_METHOD(aclmdlDestroyDataset, aclError, const aclmdlDataset *)
ORIGIN_METHOD(aclmdlDestroyDesc, aclError, aclmdlDesc *)
ORIGIN_METHOD(aclmdlExecute, aclError, uint32_t, const aclmdlDataset *, aclmdlDataset *)
ORIGIN_METHOD(aclmdlFinalizeDump, aclError)
ORIGIN_METHOD(aclmdlGetCurOutputDims, aclError, const aclmdlDesc *, size_t, aclmdlIODims *)
ORIGIN_METHOD(aclmdlGetDatasetBuffer, aclDataBuffer *, const aclmdlDataset *, size_t)
ORIGIN_METHOD(aclmdlGetDatasetNumBuffers, size_t, const aclmdlDataset *)
ORIGIN_METHOD(aclmdlGetDesc, aclError, aclmdlDesc *, uint32_t)
ORIGIN_METHOD(aclmdlGetInputDataType, aclDataType, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetInputDims, aclError, const aclmdlDesc *, size_t, aclmdlIODims *)
ORIGIN_METHOD(aclmdlGetInputIndexByName, aclError, const aclmdlDesc *, const char *, size_t *)
ORIGIN_METHOD(aclmdlGetInputNameByIndex, const char *, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetInputSizeByIndex, size_t, aclmdlDesc *, size_t index)
ORIGIN_METHOD(aclmdlGetNumInputs, size_t, aclmdlDesc *)
ORIGIN_METHOD(aclmdlGetNumOutputs, size_t, aclmdlDesc *)
ORIGIN_METHOD(aclmdlGetOutputDataType, aclDataType, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetOutputDims, aclError, const aclmdlDesc *, size_t, aclmdlIODims *)
ORIGIN_METHOD(aclmdlGetOutputNameByIndex, const char *, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetOutputSizeByIndex, size_t, aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlInitDump, aclError)
ORIGIN_METHOD(aclmdlLoadFromMem, aclError, const void *, size_t, uint32_t *)
ORIGIN_METHOD(aclmdlSetDump, aclError, const char *)
ORIGIN_METHOD(aclmdlSetDynamicBatchSize, aclError, uint32_t, aclmdlDataset *, size_t, uint64_t)
ORIGIN_METHOD(aclmdlUnload, aclError, uint32_t)
ORIGIN_METHOD(aclmdlQuerySizeFromMem, aclError, const void *, size_t, size_t *, size_t *)
ORIGIN_METHOD(aclmdlBundleGetModelId, aclError, uint32_t, size_t, uint32_t *)
ORIGIN_METHOD(aclmdlBundleLoadFromMem, aclError, const void *, size_t, uint32_t *)
ORIGIN_METHOD(aclmdlBundleUnload, aclError, uint32_t)
ORIGIN_METHOD(aclmdlLoadFromMemWithMem, aclError, const void *, size_t, uint32_t *, void *, size_t, void *, size_t)
ORIGIN_METHOD(aclmdlSetDatasetTensorDesc, aclError, aclmdlDataset *, aclTensorDesc *, size_t)
ORIGIN_METHOD(aclmdlGetInputFormat, aclFormat, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetDatasetTensorDesc, aclTensorDesc *, const aclmdlDataset *, size_t)
ORIGIN_METHOD(aclmdlSetInputDynamicDims, aclError, uint32_t, aclmdlDataset *, size_t, const aclmdlIODims *)
ORIGIN_METHOD(aclmdlGetOutputFormat, aclFormat, const aclmdlDesc *, size_t)
ORIGIN_METHOD(aclmdlGetInputDimsV2, aclError, const aclmdlDesc *, size_t, aclmdlIODims *)
ORIGIN_METHOD(aclmdlGetDynamicHW, aclError, const aclmdlDesc *, size_t, aclmdlHW *)
ORIGIN_METHOD(aclmdlGetInputDynamicDims, aclError, const aclmdlDesc *, size_t, aclmdlIODims *, size_t)
ORIGIN_METHOD(aclmdlGetInputDynamicGearCount, aclError, const aclmdlDesc *, size_t, size_t *)
ORIGIN_METHOD(aclmdlGetDynamicBatch, aclError, const aclmdlDesc *, aclmdlBatch *)
ORIGIN_METHOD(aclmdlSetDynamicHWSize, aclError, uint32_t, aclmdlDataset *, size_t, uint64_t, uint64_t)

extern aclmdlAddDatasetBufferFunObj aclmdlAddDatasetBuffer_;
extern aclmdlCreateDatasetFunObj aclmdlCreateDataset_;
extern aclmdlCreateDescFunObj aclmdlCreateDesc_;
extern aclmdlDestroyDatasetFunObj aclmdlDestroyDataset_;
extern aclmdlDestroyDescFunObj aclmdlDestroyDesc_;
extern aclmdlExecuteFunObj aclmdlExecute_;
extern aclmdlFinalizeDumpFunObj aclmdlFinalizeDump_;
extern aclmdlGetCurOutputDimsFunObj aclmdlGetCurOutputDims_;
extern aclmdlGetDatasetBufferFunObj aclmdlGetDatasetBuffer_;
extern aclmdlGetDatasetNumBuffersFunObj aclmdlGetDatasetNumBuffers_;
extern aclmdlGetDescFunObj aclmdlGetDesc_;
extern aclmdlGetInputDataTypeFunObj aclmdlGetInputDataType_;
extern aclmdlGetInputDimsFunObj aclmdlGetInputDims_;
extern aclmdlGetInputIndexByNameFunObj aclmdlGetInputIndexByName_;
extern aclmdlGetInputNameByIndexFunObj aclmdlGetInputNameByIndex_;
extern aclmdlGetInputSizeByIndexFunObj aclmdlGetInputSizeByIndex_;
extern aclmdlGetNumInputsFunObj aclmdlGetNumInputs_;
extern aclmdlGetNumOutputsFunObj aclmdlGetNumOutputs_;
extern aclmdlGetOutputDataTypeFunObj aclmdlGetOutputDataType_;
extern aclmdlGetOutputDimsFunObj aclmdlGetOutputDims_;
extern aclmdlGetOutputNameByIndexFunObj aclmdlGetOutputNameByIndex_;
extern aclmdlGetOutputSizeByIndexFunObj aclmdlGetOutputSizeByIndex_;
extern aclmdlInitDumpFunObj aclmdlInitDump_;
extern aclmdlLoadFromMemFunObj aclmdlLoadFromMem_;
extern aclmdlSetDumpFunObj aclmdlSetDump_;
extern aclmdlSetDynamicBatchSizeFunObj aclmdlSetDynamicBatchSize_;
extern aclmdlUnloadFunObj aclmdlUnload_;
extern aclmdlQuerySizeFromMemFunObj aclmdlQuerySizeFromMem_;
extern aclmdlBundleGetModelIdFunObj aclmdlBundleGetModelId_;
extern aclmdlBundleLoadFromMemFunObj aclmdlBundleLoadFromMem_;
extern aclmdlBundleUnloadFunObj aclmdlBundleUnload_;
extern aclmdlLoadFromMemWithMemFunObj aclmdlLoadFromMemWithMem_;
extern aclmdlSetDatasetTensorDescFunObj aclmdlSetDatasetTensorDesc_;
extern aclmdlGetInputFormatFunObj aclmdlGetInputFormat_;
extern aclmdlGetDatasetTensorDescFunObj aclmdlGetDatasetTensorDesc_;
extern aclmdlSetInputDynamicDimsFunObj aclmdlSetInputDynamicDims_;
extern aclmdlGetOutputFormatFunObj aclmdlGetOutputFormat_;
extern aclmdlGetInputDimsV2FunObj aclmdlGetInputDimsV2_;
extern aclmdlGetDynamicHWFunObj aclmdlGetDynamicHW_;
extern aclmdlGetInputDynamicDimsFunObj aclmdlGetInputDynamicDims_;
extern aclmdlGetInputDynamicGearCountFunObj aclmdlGetInputDynamicGearCount_;
extern aclmdlGetDynamicBatchFunObj aclmdlGetDynamicBatch_;
extern aclmdlSetDynamicHWSizeFunObj aclmdlSetDynamicHWSize_;

void LoadAclMdlApiSymbol(const std::string &ascend_path);
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_MDL_SYMBOL_H_
