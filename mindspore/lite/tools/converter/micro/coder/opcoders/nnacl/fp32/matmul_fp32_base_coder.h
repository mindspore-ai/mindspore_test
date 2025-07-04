/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_MATMUL_FP32_BASE_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_MATMUL_FP32_BASE_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/op_coder.h"
#include "nnacl/matmul_parameter.h"
#include "wrapper/base/micro_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
class MatMulFP32BaseCoder : public OperatorCoder {
 public:
  MatMulFP32BaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~MatMulFP32BaseCoder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  virtual int ReSize();

 protected:
  void ResizeParameter();
  virtual int InitBufferForBias();
  virtual int InitBufferA();
  virtual int InitBufferB();
  virtual int CollectFilesForTarget(CoderContext *const context);
  virtual int Init();
  virtual void InitParameter();
  void CalculateOutBatchSize();

 private:
  int InitMatrixA(const float *src_ptr);
  int InitMatrixB(const float *src_ptr);
  void GenerateMatrixCalculation(nnacl::NNaclFp32Serializer *const code, const std::string *c_str,
                                 const std::string *a_pack_str, const std::string *b_pack_str, int cur_oc);

 protected:
  Tensor *filter_tensor_{nullptr};
  Tensor *bias_tensor_{nullptr};
  MicroMatmulParameter params_;
  void *a_pack_ptr_ = nullptr;
  void *b_pack_ptr_ = nullptr;
  void *bias_ptr_{nullptr};
  bool vec_matmul_{false};
  bool de_quant_flag_{false};
  bool a_packed_{false};
  bool b_packed_{false};
  int col_tile_{0};
  int row_tile_{0};
  int thread_stride_{0};
  int thread_count_{0};
  size_t bias_pack_ptr_size_{0};
  size_t ori_bias_pack_ptr_size_{0};
  size_t a_pack_ptr_size_{0};
  size_t b_pack_ptr_size_{0};
  bool is_bias_broadcast_{false};
  TypeId data_type_{kNumberTypeFloat32};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_MATMUL_FP32_BASE_CODER_H_
