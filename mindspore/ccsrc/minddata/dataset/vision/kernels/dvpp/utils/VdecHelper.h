/**
# Adapted from
# https://gitee.com/ascend/samples/blob/master/cplusplus/common/acllite/include/VdecHelper.h
*
* @file VdecHelper.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_VDEC_HELPER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_VDEC_HELPER_H_

#include <cstdint>
#include <iostream>
#include <memory>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "minddata/dataset/vision/kernels/dvpp/utils/AclLiteError.h"
#include "minddata/dataset/vision/kernels/dvpp/utils/AclLiteType.h"

class VdecHelper {
 public:
  VdecHelper(int channel, uint32_t width, uint32_t height, int type, aclvdecCallback callback,
             uint32_t outFormat = PIXEL_FORMAT_YUV_SEMIPLANAR_420);
  ~VdecHelper();

  static void *SubscribeReportThreadFunc(void *arg);

  AclLiteError Init();
  void DestroyResource();
  void DestroyChannel();

  AclLiteError Process(const std::shared_ptr<FrameData> &frameData, void *userData);
  AclLiteError SetFormat(uint32_t format);
  AclLiteError VideoParamCheck() const;
  bool IsExit() const { return isExit_; }
  aclrtContext GetContext() { return context_; }

 private:
  AclLiteError CreateVdecChannelDesc();
  AclLiteError CreateInputStreamDesc(const std::shared_ptr<FrameData> &frameData);
  AclLiteError CreateOutputPicDesc(size_t size);
  void UnsubscribReportThread();

 private:
  int channelId_;

  /* 1：YUV420 semi-planner（nv12）
     2：YVU420 semi-planner（nv21）
  */
  uint32_t format_;

  /* 0：H265 main level
   * 1：H264 baseline level
   * 2：H264 main level
   * 3：H264 high level
   */
  uint32_t enType_;

  uint32_t frameWidth_;
  uint32_t frameHeight_;
  uint32_t alignWidth_;
  uint32_t alignHeight_;
  uint32_t outputPicSize_;
  void *outputPicBuf_;
  aclvdecCallback callback_;
  aclrtContext context_{};
  aclrtStream stream_{};

  aclvdecChannelDesc *vdecChannelDesc_;
  acldvppStreamDesc *inputStreamDesc_;
  acldvppPicDesc *outputPicDesc_;

  pthread_t subscribeThreadId_{};
  bool isExit_;
  bool isReleased_;
  bool isChannelExit_;
};

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_VDEC_HELPER_H_
