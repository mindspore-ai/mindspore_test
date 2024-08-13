/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ATB_TRAINOPPARAM_H
#define ATB_TRAINOPPARAM_H
#include <cstdint>
#include <string>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file train_op_params.h
//!
//! \brief 定义加速库所有训练算子参数
//!

namespace atb {

namespace train {

//!
//! \struct GenAttentionMaskParam
//!
//! \brief 将attentionMask根据每个batch的实际seqlen进行转化，得到结果为一维tensor。
//!
struct GenAttentionMaskParam {
    //! \brief 多头注意力机制的head数
    int32_t headNum = 1;
    //! \brief 存储unpad场景下每个batch实际seqlen的值。元素个数为batchSize，最大为32。
    atb::SVector<int32_t> seqLen;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const GenAttentionMaskParam &left, const GenAttentionMaskParam &right)
{
    return left.headNum == right.headNum && left.seqLen == right.seqLen;
}

//!
//! \struct RopeGradParam
//!
//! \brief 旋转位置编码处理的反向。
//!
struct RopeGradParam {
    //! \brief 存储unpad场景下每个batch实际qSseqlen的值。size不能为0
    std::vector<int32_t> qSeqLen;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//!
//! \return bool值
//!
inline bool operator==(const RopeGradParam &left, const RopeGradParam &right)
{
    return left.qSeqLen == right.qSeqLen;
}

//!
//! \struct FastSoftMaxParam
//!
//! \brief 将unpad处理后的Q矩阵和K矩阵相乘的结果做Softmax处理。
//!
//! \warning seqLen数组长度不超过32，且要求各元素大于0。
//!
struct FastSoftMaxParam {
    //! \brief Attention的head数量。
    int32_t headNum = 0;
    //! \brief 每个batch的实际输入长度。元素个数为batchSize，最大不超过32
    std::vector<int32_t> qSeqLen;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//!
//! \return bool值
//!
inline bool operator==(const FastSoftMaxParam &left, const FastSoftMaxParam &right)
{
    return left.headNum == right.headNum && left.qSeqLen == right.qSeqLen;
}

//!
//! \struct FastSoftMaxGradParam
//!
//! \brief 将unpad处理后的Q矩阵和K矩阵相乘的结果做Softmax的反向计算处理。
//!
//! \warning seqLen数组长度不超过32，且要求各元素大于0。
//!
struct FastSoftMaxGradParam {
    //! \brief Attention的head数量。
    int32_t headNum = 0;
    //! \brief 每个batch的实际输入长度。元素个数为batchSize，最大不超过32
    std::vector<int32_t> qSeqLen;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool值
//!
inline bool operator==(const FastSoftMaxGradParam &left, const FastSoftMaxGradParam &right)
{
    return left.headNum == right.headNum && left.qSeqLen == right.qSeqLen;
}

//!
//! \struct StridedBatchMatmulParam
//!
//! \brief 对矩阵进行分组，指定每组矩阵之间的步长，实现更加灵活的矩阵乘法操作。
//!
struct StridedBatchMatmulParam {
    //! \brief 是否转置A矩阵。
    bool transposeA = false;
    //! \brief 是否转置B矩阵。
    bool transposeB = false;
    //! \brief batch个数batchSize。
    int32_t batch = 1;
    //! \brief 多头注意力机制的head数。
    int32_t headNum = 1;
    //! \brief A矩阵参与一次矩阵乘指令的shape大小。元素个数为batchSize。
    std::vector<int32_t> m;
    //! \brief B矩阵参与一次矩阵乘指令的shape大小。元素个数为batchSize。
    std::vector<int32_t> n;
    //! \brief C矩阵参与一次矩阵乘指令的shape大小。元素个数为batchSize。
    std::vector<int32_t> k;
    //! \brief 表示矩阵A的列数。元素个数为batchSize。
    std::vector<int32_t> lda;
    //! \brief 表示矩阵B的列数。元素个数为batchSize。
    std::vector<int32_t> ldb;
    //! \brief 表示矩阵C的列数。元素个数为batchSize。
    std::vector<int32_t> ldc;
    //! \brief 矩阵A在内存中相邻两次计算之间的跨度。元素个数为batchSize。
    std::vector<int32_t> strideA;
    //! \brief 矩阵B在内存中相邻两次计算之间的跨度。元素个数为batchSize。
    std::vector<int32_t> strideB;
    //! \brief Descript矩阵C在内存中相邻两次计算之间的跨度。元素个数为batchSize。
    std::vector<int32_t> strideC;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool值
//!
inline bool operator==(const StridedBatchMatmulParam &left, const StridedBatchMatmulParam &right)
{
    return left.transposeA == right.transposeA && left.transposeB == right.transposeB &&
    left.batch == right.batch && left.headNum == right.headNum && left.m == right.m &&
    left.n == right.n && left.k == right.k && left.lda == right.lda && left.ldb == right.ldb &&
    left.ldc == right.ldc && left.strideA == right.strideA && left.strideB == right.strideB &&
    left.strideC == right.strideC;
}
//!
//! \struct UnpadWithHiddenStateParam
//!
//! \brief 在llama的微调场景中的unpad方案调用算子，Unpad Transformer的encoder过程中，序列长度不再按照最大长度计算，根据实际的长度进行计算（向上pad到16倍数），减少计算量。
//!
struct UnpadWithHiddenStateParam {
    //! \brief 每个batch实际qSeqlen的值，元素个数为batchSize。batchSize的值最大不超过32。
    std::vector<int32_t> qSeqLen;
    //! \brief 最大qSeqLen值。取值不超过4096。
    int32_t maxSeqLen = 4096;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool值
//!
inline bool operator==(const UnpadWithHiddenStateParam &left, const UnpadWithHiddenStateParam &right)
{
    return left.qSeqLen == right.qSeqLen && left.maxSeqLen == right.maxSeqLen;
}

//!
//! \struct PadWithHiddenStateParam
//!
//! \brief 在llama的微调场景中的unpad方案调用算子，Transformer的encoder过程中，序列长度不再按照最大长度计算，根据实际的长度进行计算（向上pad到16倍数），减少计算量。
//!
struct PadWithHiddenStateParam {
    //! \brief 每个batch的实际输入长度。元素个数为batchSize。batchSize的值最大不超过32。
    std::vector<int32_t> qSeqLen;
    //! \brief qSeqLen中最大输入长度。取值不超过4096。
    int32_t maxSeqLen = 4096;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool值
//!
inline bool operator==(const PadWithHiddenStateParam &left, const PadWithHiddenStateParam &right)
{
    return left.qSeqLen == right.qSeqLen && left.maxSeqLen == right.maxSeqLen;
}

//!
//! \struct RmsNormBackwardParam
//!
//! \brief rmsnorm 的反向计算。
//!
struct RmsNormBackwardParam {};

}  // namespace train
}  // namespace atb_train
#endif