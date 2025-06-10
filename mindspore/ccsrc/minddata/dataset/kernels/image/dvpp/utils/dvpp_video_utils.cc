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
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_video_utils.h"

#include <vector>
#include <map>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/prctl.h>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t MAX_CHN_HEIGHT = 4096;
constexpr uint32_t MAX_CHN_WIDTH = 4096;
constexpr int32_t SEND_TIMEOUT = 30;
constexpr uint32_t WAIT_TIMEOUT = 5000000;  // 5000000us
constexpr uint32_t REF_FRAME_NUM = 16;
constexpr uint32_t DISPLAY_FRAME_NUM = 16;
constexpr uint32_t FRAME_BUF_CNT = REF_FRAME_NUM + DISPLAY_FRAME_NUM + 1;

pthread_t g_vdec_get_thread[VDEC_MAX_CHNL_NUM] = {0};
uint32_t g_get_exit_state[VDEC_MAX_CHNL_NUM] = {0};
std::vector<std::vector<std::shared_ptr<DeviceBuffer>>> g_out_queue(VDEC_MAX_CHNL_NUM);  // save success decoded frame
std::mutex outTensorMapMutex[VDEC_MAX_CHNL_NUM];                                         // map is not Thread-safe
std::map<hi_u64, std::shared_ptr<DeviceBuffer>> outTensorMap[VDEC_MAX_CHNL_NUM];

struct GetThreadPara {
  uint32_t chnId;
  uint32_t deviceId;
  uint32_t totalFrame;
  uint32_t successCnt;
};

GetThreadPara g_getPara[VDEC_MAX_CHNL_NUM];

static inline bool ValidChnNum(uint32_t chn) { return (chn < VDEC_MAX_CHNL_NUM); }

static inline void get_current_time_us(uint64_t &timeUs) {
  struct timeval curTime;
  gettimeofday(&curTime, NULL);
  timeUs = static_cast<uint64_t>(curTime.tv_sec) * 1000000 + curTime.tv_usec;  // 1s = 1000000 us
}

template <class T>
static inline void LoadFunc(void *const handle, T &funPtr, const std::string &funName) {
  funPtr = reinterpret_cast<T>(dlsym(handle, funName.c_str()));
  if (funPtr == nullptr) {
    MS_EXCEPTION(RuntimeError) << "vdec function not load, func name " << funName.c_str();
  }
}

VideoDecoder &VideoDecoder::GetInstance() {
  static VideoDecoder instance;
  return instance;
}

VideoDecoder::VideoDecoder() {
  // LoadFunctions();
  for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
    channelStatus_[i] = ChnStatus::DESTROYED;
  }
}

VideoDecoder::~VideoDecoder() {}

int32_t VideoDecoder::GetUnusedChn(uint32_t &chn) {
  for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
    const std::lock_guard<std::mutex> guard(channelMutex_[i]);
    if (channelStatus_[i] != ChnStatus::DESTROYED) {
      continue;
    } else {
      channelStatus_[i] = ChnStatus::CREATED;
      chn = i;
      return 0;
    }
  }
  return -1;
}

void VideoDecoder::PutChn(uint32_t chn) {
  const std::lock_guard<std::mutex> guard(channelMutex_[chn]);
  channelStatus_[chn] = ChnStatus::DESTROYED;
}

bool VideoDecoder::ChannelCreated(uint32_t chn) {
  const std::lock_guard<std::mutex> guard(channelMutex_[chn]);
  return (channelStatus_[chn] == ChnStatus::CREATED);
}

hi_s32 VideoDecoder::sys_init(void) {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Get ms context failed by MsContext::GetInstance()";
  }
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  if (device_context_ == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Get device context failed by ms context";
  }
  device_context_->Initialize();
  if (device_context_->device_res_manager_ == nullptr) {
    MS_EXCEPTION(RuntimeError) << "The device resource manager is null";
  }

  std::string soc_version;
  if (GetSocName(&soc_version) != APP_ERR_OK) {
    MS_EXCEPTION(RuntimeError) << "Get Soc Version failed.";
  }
  if (soc_version.find("Ascend910B") == std::string::npos && soc_version.find("Ascend910_93") == std::string::npos) {
    // reset the device_context_, because the Soc is not support yet
    device_context_ = nullptr;
    MS_EXCEPTION(RuntimeError) << "The SoC: " << soc_version << " is not Ascend910B / Ascend910_93";
  }
  auto ret = hi_mpi_sys_init();
  return ret;
}

hi_s32 VideoDecoder::sys_exit(void) { return hi_mpi_sys_exit(); }

hi_u32 VideoDecoder::get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr) {
  return hi_vdec_get_pic_buf_size(type, buf_attr);
}

hi_u32 VideoDecoder::get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height) {
  return hi_vdec_get_tmv_buf_size(type, width, height);
}

hi_s32 VideoDecoder::create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr) {
  auto ret = hi_mpi_vdec_create_chn(chn, attr);
  return ret;
}

hi_s32 VideoDecoder::destroy_chn(hi_vdec_chn chn) { return hi_mpi_vdec_destroy_chn(chn); }

hi_s32 VideoDecoder::sys_set_chn_csc_matrix(hi_vdec_chn chn) {
  return hi_mpi_sys_set_chn_csc_matrix(HI_ID_VDEC, chn, HI_CSC_MATRIX_BT601_NARROW, nullptr);
}

hi_s32 VideoDecoder::start_recv_stream(hi_vdec_chn chn) { return hi_mpi_vdec_start_recv_stream(chn); }

hi_s32 VideoDecoder::stop_recv_stream(hi_vdec_chn chn) { return hi_mpi_vdec_stop_recv_stream(chn); }

hi_s32 VideoDecoder::query_status(hi_vdec_chn chn, hi_vdec_chn_status *status) {
  return hi_mpi_vdec_query_status(chn, status);
}

hi_s32 VideoDecoder::reset_chn(hi_vdec_chn chn) { return hi_mpi_vdec_reset_chn(chn); }

hi_s32 VideoDecoder::send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream, hi_vdec_pic_info *vdec_pic_info,
                                 hi_s32 milli_sec) {
  return hi_mpi_vdec_send_stream(chn, stream, vdec_pic_info, milli_sec);
}

hi_s32 VideoDecoder::get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info, hi_vdec_supplement_info *supplement,
                               hi_vdec_stream *stream, hi_s32 milli_sec) {
  return hi_mpi_vdec_get_frame(chn, frame_info, supplement, stream, milli_sec);
}

hi_s32 VideoDecoder::release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info) {
  return hi_mpi_vdec_release_frame(chn, frame_info);
}

hi_s32 VideoDecoder::dvpp_malloc(hi_u32 dev_id, hi_void **dev_ptr, hi_u64 size) {
  return hi_mpi_dvpp_malloc(dev_id, dev_ptr, size);
}

hi_s32 VideoDecoder::dvpp_free(hi_void *dev_ptr) { return hi_mpi_dvpp_free(dev_ptr); }

static void vdec_reset_chn(uint32_t chn) {
  int32_t ret = VideoDecoder::GetInstance().stop_recv_stream(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "reset chn " << chn << ", hi_mpi_vdec_stop_recv_stream failed, ret = " << ret;
  }

  ret = VideoDecoder::GetInstance().reset_chn(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "reset chn " << chn << ", hi_mpi_vdec_reset_chn failed, ret = " << ret;
  }

  ret = VideoDecoder::GetInstance().start_recv_stream(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "reset chn " << chn << ", hi_mpi_vdec_start_recv_stream failed, ret = " << ret;
  }
}

static thread_local int32_t local_device = -1;
static std::unordered_map<int32_t, aclrtContext> used_devices;
std::recursive_mutex mtx;

aclError SetDevice(int32_t device) {
  if (device < 0) {
    MS_EXCEPTION(RuntimeError) << "Device id must be positive!";
  }

  if (local_device == device) {
    return ACL_SUCCESS;
  }

  aclError err = aclrtSetDevice(device);
  if (err == ACL_SUCCESS) {
    local_device = device;
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(local_device) == used_devices.end()) {
      aclError ret = aclrtGetCurrentContext(&used_devices[local_device]);
      if (ret != ACL_SUCCESS) {
        MS_EXCEPTION(RuntimeError) << "Call aclrtGetCurrentContext failed, ret: " << ret;
      }
    }
  }
  return err;
}

void *get_pic(void *args) {
  prctl(PR_SET_NAME, "VdecGetPic", 0, 0, 0);
  GetThreadPara *para = (GetThreadPara *)args;
  uint32_t chanId = para->chnId;
  aclError device_ret = SetDevice(para->deviceId);
  if (device_ret != ACL_SUCCESS) {
    MS_EXCEPTION(RuntimeError) << "Set device failed, ret: " << device_ret;
  }

  int32_t ret = HI_SUCCESS;
  hi_video_frame_info frame{};
  hi_vdec_stream stream{};
  int32_t decResult = 0;  // Decode result
  hi_u64 outputBuffer = 0;
  uint32_t successCnt = 0;
  uint32_t failCnt = 0;
  int32_t timeOut = 0;

  auto outQueue = std::vector<std::shared_ptr<DeviceBuffer>>(para->totalFrame);
  g_get_exit_state[chanId] = 0;

  while (g_get_exit_state[chanId] == 0) {
    ret = VideoDecoder::GetInstance().get_frame(chanId, &frame, nullptr, &stream, timeOut);
    if (ret == HI_SUCCESS) {
      // Flush decode end time
      outputBuffer = static_cast<hi_u64>(reinterpret_cast<uintptr_t>(frame.v_frame.virt_addr[0]));
      decResult = frame.v_frame.frame_flag;
      if (decResult == 0) {  // 0: Decode success
        const std::lock_guard<std::mutex> guard(outTensorMapMutex[chanId]);
        auto iter = outTensorMap[chanId].find(outputBuffer);
        if (iter != outTensorMap[chanId].end()) {
          outQueue[successCnt] = iter->second;
          outTensorMap[chanId].erase(iter);
          successCnt++;
        }
      } else if (decResult == 1) {  // 1: Decode fail
        failCnt++;
        MS_LOG(WARNING) << "chn " << chanId << " GetFrame Success, decode failed, fail count " << failCnt;
      } else if (decResult == 2) {
        // 2:This result is returned for the second field of
        // the interlaced field stream, which is normal.
      } else if (decResult == 3) {  // 3: Reference frame number set error
        failCnt++;
        MS_LOG(WARNING) << "chn " << chanId << " GetFrame Success, refFrame num Error, fail count " << failCnt;
      } else if (decResult == 4) {  // 4: Reference frame size set error
        failCnt++;
        MS_LOG(WARNING) << "chn " << chanId << " GetFrame Success, refFrame Size Error, fail count " << failCnt;
      }
      // Release Frame
      ret = VideoDecoder::GetInstance().release_frame(chanId, &frame);
      if (ret != 0) {
        MS_EXCEPTION(RuntimeError) << "chn " << chanId << ", hi_mpi_vdec_release_frame failed, ret = " << ret;
      }
    } else {
      // 500us
      usleep(500);
    }
  }

  g_out_queue[chanId] = outQueue;
  para->successCnt = successCnt;
  return (void *)HI_SUCCESS;
}

int64_t dvpp_sys_init() { return static_cast<int64_t>(VideoDecoder::GetInstance().sys_init()); }

int64_t dvpp_sys_exit() { return static_cast<int64_t>(VideoDecoder::GetInstance().sys_exit()); }

int64_t dvpp_vdec_create_chnl(int64_t pType) {
  if (pType != 96 && pType != 265) {  // H264:96 H265:265
    MS_EXCEPTION(RuntimeError) << "invalid pType " << pType << ", should be H264:96, H265:265";
  }
  uint32_t chn = 0;
  int32_t ret = VideoDecoder::GetInstance().GetUnusedChn(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "get unused chn failed";
  }

  hi_vdec_chn_attr chnAttr{};
  chnAttr.type = static_cast<hi_payload_type>(pType);
  chnAttr.mode = HI_VDEC_SEND_MODE_FRAME;  // Only support frame mode
  chnAttr.pic_width = MAX_CHN_WIDTH;
  chnAttr.pic_height = MAX_CHN_HEIGHT;
  chnAttr.stream_buf_size = MAX_CHN_WIDTH * MAX_CHN_HEIGHT * 3 / 2;
  chnAttr.frame_buf_cnt = FRAME_BUF_CNT;
  hi_pic_buf_attr buf_attr{
    MAX_CHN_WIDTH, MAX_CHN_HEIGHT, 0, HI_DATA_BIT_WIDTH_10, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420, HI_COMPRESS_MODE_NONE};
  chnAttr.frame_buf_size = VideoDecoder::GetInstance().get_pic_buf_size(chnAttr.type, &buf_attr);
  chnAttr.video_attr.ref_frame_num = REF_FRAME_NUM;
  chnAttr.video_attr.temporal_mvp_en = HI_TRUE;
  chnAttr.video_attr.tmv_buf_size =
    VideoDecoder::GetInstance().get_tmv_buf_size(chnAttr.type, MAX_CHN_WIDTH, MAX_CHN_HEIGHT);

  ret = VideoDecoder::GetInstance().create_chn(chn, &chnAttr);
  if (ret != HI_SUCCESS) {
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "hi_mpi_vdec_create_chn " << chn << " failed, ret = " << ret;
    return -1;
  }

  ret = VideoDecoder::GetInstance().sys_set_chn_csc_matrix(chn);
  if (ret != HI_SUCCESS) {
    (void)VideoDecoder::GetInstance().destroy_chn(chn);
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "chn " << chn << ", hi_mpi_sys_set_chn_csc_matrix failed, ret = " << ret;
    return -1;
  }

  ret = VideoDecoder::GetInstance().start_recv_stream(chn);
  if (ret != HI_SUCCESS) {
    (void)VideoDecoder::GetInstance().destroy_chn(chn);
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "chn " << chn << ", hi_mpi_vdec_start_recv_stream failed, ret = " << ret;
    return -1;
  }

  return static_cast<int64_t>(chn);
}

aclError GetDevice(int32_t *device) {
  if (local_device >= 0) {
    *device = local_device;
    return ACL_SUCCESS;
  }
  aclError err = aclrtGetDevice(device);
  if (err != ACL_SUCCESS) {
    MS_EXCEPTION(RuntimeError) << "Call aclrtGetDevice failed, ret: " << err;
  }
  if (err == ACL_SUCCESS) {
    local_device = *device;
  } else if (err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_SUCCESS) {
    *device = 0;
    local_device = 0;
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(local_device) == used_devices.end()) {
      auto ret = aclrtGetCurrentContext(&used_devices[local_device]);
      if (ret != ACL_SUCCESS) {
        MS_EXCEPTION(RuntimeError) << "Call aclrtGetCurrentContext failed, ret: " << ret;
      }
    }
    return ACL_SUCCESS;
  }
  return err;
}

int64_t dvpp_vdec_start_get_frame(int64_t chnId, int64_t totalFrame) {
  if (!ValidChnNum(chnId)) {
    MS_EXCEPTION(RuntimeError) << "invalid chn " << chnId;
  }

  int32_t deviceId = 0;
  aclError aclRet = GetDevice(&deviceId);
  if (aclRet != 0) {
    MS_EXCEPTION(RuntimeError) << "get device id failed, ret = " << aclRet;
  }

  g_getPara[chnId].chnId = chnId;
  g_getPara[chnId].deviceId = deviceId;
  g_getPara[chnId].totalFrame = totalFrame;
  g_getPara[chnId].successCnt = 0;
  g_vdec_get_thread[chnId] = 0;
  int32_t ret = pthread_create(&g_vdec_get_thread[chnId], 0, get_pic, static_cast<void *>(&g_getPara[chnId]));
  if (ret != 0) {
    g_vdec_get_thread[chnId] = 0;
    MS_EXCEPTION(RuntimeError) << "Chn " << chnId << ", create get pic thread failed, ret = " << ret;
    return -1;
  }

  return 0;
}

int64_t dvpp_vdec_send_stream(int64_t chnId, const std::shared_ptr<Tensor> &input, int64_t outFormat, bool display,
                              std::shared_ptr<DeviceBuffer> *out) {
  if (!ValidChnNum(chnId)) {
    MS_EXCEPTION(RuntimeError) << "invalid chn " << chnId;
  }
  hi_pixel_format outputFormat = static_cast<hi_pixel_format>(outFormat);
  if (outputFormat != HI_PIXEL_FORMAT_RGB_888 && outputFormat != HI_PIXEL_FORMAT_BGR_888 &&
      outputFormat != HI_PIXEL_FORMAT_RGB_888_PLANAR && outputFormat != HI_PIXEL_FORMAT_BGR_888_PLANAR) {
    MS_EXCEPTION(RuntimeError) << "invalid outFormat " << outputFormat << ", should be " << HI_PIXEL_FORMAT_RGB_888
                               << " or " << HI_PIXEL_FORMAT_BGR_888 << " or " << HI_PIXEL_FORMAT_RGB_888_PLANAR
                               << " or " << HI_PIXEL_FORMAT_BGR_888_PLANAR;
  }

  void *device_address =
    VideoDecoder::GetInstance().device_context_->device_res_manager_->AllocateMemory(input->SizeInBytes());
  if (device_address == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Allocate device memory failed.";
  }
  VideoDecoder::GetInstance().device_context_->device_res_manager_->SwapIn(
    reinterpret_cast<void *>(input->GetMutableBuffer()), device_address, input->SizeInBytes(), nullptr);

  int64_t selfSizeBytes = input->SizeInBytes();

  int64_t outSizeBytes = (*out)->GetBufferSize();

  hi_vdec_stream stream{};
  uint64_t currentSendTime = 0;
  get_current_time_us(currentSendTime);
  stream.pts = currentSendTime;
  stream.addr = static_cast<hi_u8 *>(device_address);
  stream.len = selfSizeBytes;
  stream.end_of_frame = HI_TRUE;
  stream.end_of_stream = HI_FALSE;
  stream.need_display = display ? HI_TRUE : HI_FALSE;

  hi_vdec_pic_info outPicInfo{};
  outPicInfo.height = 0;
  outPicInfo.width = 0;
  outPicInfo.width_stride = 0;
  outPicInfo.height_stride = 0;
  outPicInfo.pixel_format = outputFormat;
  outPicInfo.vir_addr = 0;
  outPicInfo.buffer_size = 0;
  if (display) {
    outPicInfo.vir_addr = static_cast<hi_u64>(reinterpret_cast<uintptr_t>((*out)->GetBuffer()));
    outPicInfo.buffer_size = outSizeBytes;
  }

  uint32_t sendOneFrameCnt = 0;
  int32_t ret = 0;
  do {
    sendOneFrameCnt++;
    // Send one frame data
    ret = VideoDecoder::GetInstance().send_stream(chnId, &stream, &outPicInfo, SEND_TIMEOUT);
    if (sendOneFrameCnt > 30) {  // if send stream timeout 30 times, end the decode process
      if (ret != 0) {
        vdec_reset_chn(chnId);
      }
      break;
    }
  } while (ret == HI_ERR_VDEC_BUF_FULL);  // Try again
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "chn " << chnId << ", hi_mpi_vdec_send_stream failed, ret = " << ret;
  }

  if (display) {
    const std::lock_guard<std::mutex> guard(outTensorMapMutex[chnId]);
    outTensorMap[chnId].insert(
      std::make_pair(static_cast<hi_u64>(reinterpret_cast<uintptr_t>((*out)->GetBuffer())), *out));
  }

  return 0;
}

std::shared_ptr<DeviceBuffer> dvpp_vdec_stop_get_frame(int64_t chnId, int64_t totalFrame) {
  hi_vdec_chn_status status{};
  hi_vdec_chn_status pre_status{};

  hi_vdec_stream stream{};
  hi_vdec_pic_info outPicInfo{};
  // Send stream end flage
  stream.addr = NULL;
  stream.len = 0;
  stream.end_of_frame = HI_FALSE;
  stream.end_of_stream = HI_TRUE;  // Stream end flage
  outPicInfo.vir_addr = 0;
  outPicInfo.buffer_size = 0;
  int32_t ret = VideoDecoder::GetInstance().send_stream(chnId, &stream, &outPicInfo, -1);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "chn " << chnId
                               << ", hi_mpi_vdec_send_stream send end_of_stream failed, ret = " << ret;
  }

  uint32_t waitTimes = 0;
  uint32_t sleepTime = 10000;  // 10000us
  ret = VideoDecoder::GetInstance().stop_recv_stream(chnId);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "chn " << chnId << ", hi_mpi_vdec_stop_recv_stream failed, ret = " << ret;
  }

  while (waitTimes < WAIT_TIMEOUT) {
    ret = VideoDecoder::GetInstance().query_status(chnId, &status);
    if (ret != 0) {
      MS_EXCEPTION(RuntimeError) << "chn " << chnId << ", hi_mpi_vdec_query_status failed, ret = " << ret;
    }
    if (((status.left_stream_bytes == 0) && (status.left_decoded_frames == 0))) {
      break;
    }
    if (status.left_decoded_frames == pre_status.left_decoded_frames) {
      waitTimes += sleepTime;
    } else {
      waitTimes = 0;
    }
    pre_status = status;
    // 10000us
    usleep(sleepTime);

    if (waitTimes >= WAIT_TIMEOUT) {
      vdec_reset_chn(chnId);
      break;
    }
  }

  g_get_exit_state[chnId] = 1;  // notify get thread exit

  ret = pthread_join(g_vdec_get_thread[chnId], nullptr);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "chn " << chnId << ", pthread_join get_pic thread failed, ret = " << ret;
  }
  g_vdec_get_thread[chnId] = 0;

  // all frame success
  if (g_getPara[chnId].successCnt == totalFrame) {
    g_out_queue[chnId].clear();
    outTensorMap[chnId].clear();
    // return at::empty({0});
    std::vector<size_t> shape = {0};
    return std::make_shared<DeviceBuffer>(shape);
  }

  // some frame failed
  std::shared_ptr<DeviceBuffer> buffer = g_out_queue[chnId][0];
  auto new_shape = buffer->GetShape();
  (void)new_shape.insert(new_shape.begin(), g_getPara[chnId].successCnt);
  auto result_tensor = std::make_shared<DeviceBuffer>(new_shape);
  for (int i = 0; i < g_getPara[chnId].successCnt; i++) {
    std::shared_ptr<DeviceBuffer> dest = std::make_shared<DeviceBuffer>(result_tensor, i);
    auto aclRet = aclrtMemcpy(dest->GetBuffer(), dest->GetBufferSize(), g_out_queue[chnId][i]->GetBuffer(),
                              g_out_queue[chnId][i]->GetBufferSize(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (aclRet != 0) {
      MS_EXCEPTION(RuntimeError) << "aclrtMemcpy failed, ret = " << aclRet;
    }
  }

  g_out_queue[chnId].clear();
  outTensorMap[chnId].clear();
  return result_tensor;
}

int64_t dvpp_vdec_destroy_chnl(int64_t chnId) {
  int32_t ret = VideoDecoder::GetInstance().destroy_chn(chnId);
  VideoDecoder::GetInstance().PutChn(chnId);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "chn " << chnId << ", hi_mpi_vdec_destroy_chn failed, ret " << ret;
  }
  return 0;
}

int64_t dvpp_malloc(uint32_t dev_id, void **dev_ptr, uint64_t size) {
  return static_cast<int64_t>(VideoDecoder::GetInstance().dvpp_malloc(dev_id, dev_ptr, size));
}

int64_t dvpp_free(void *dev_ptr) { return static_cast<int64_t>(VideoDecoder::GetInstance().dvpp_free(dev_ptr)); }

int64_t dvpp_memcpy(const std::shared_ptr<DeviceBuffer> &src, void *dest) {
  return aclrtMemcpy(dest, src->GetBufferSize(), src->GetBuffer(), src->GetBufferSize(), ACL_MEMCPY_DEVICE_TO_HOST);
}
}  // namespace dataset
}  // namespace mindspore
