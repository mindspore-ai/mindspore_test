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
#ifndef ATB_INFEROPPARAM_H
#define ATB_INFEROPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file infer_op_params.h
//!
//! \brief 定义加速库所有推理算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace infer {

//!
//! \enum QuantType
//!
//! \brief 量化支持的类型
//!
enum QuantType : int {
    QUANT_UNDEFINED = 0, //!< 不量化
    QUANT_INT4,          //!< 当前不支持
    QUANT_INT8,          //!< int8量化
    QUANT_INT16,         //!< 当前不支持
    QUANT_FLOAT8,        //!< 当前不支持
    QUANT_FLOAT16,       //!< 当前不支持
};

//!
//! \enum DynamicQuantType
//!
//! \brief 动态量化支持的类型
//!
enum DynamicQuantType : int {
    DYNAMIC_QUANT_UNDEFINED = 0, //!< 非动态量化
    DYNAMIC_QUANT_SYMMETRIC,     //!< 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC,    //!< 非对称动态量化，暂不支持
};

//!
//! \enum ActivationType
//!
//! \brief 激活支持的类型
//!
//! ACTIVATION_SWIGLU_FORWARD: Atlas 300I DUO中只支持32位对齐的数据、Atlas 300I DUO中不支持bfloat16类型数据
//! ACTIVATION_SWIGLU_BACKWARD: 只支持Atlas 800I A2
//! ACTIVATION_GELU: bf16只支持Atlas 800I A2
//!
enum ActivationType : int {
    ACTIVATION_UNDEFINED = 0,   //!< 未定义
    ACTIVATION_RELU,            //!< RELU激活类型
    ACTIVATION_GELU,            //!< GELU激活类型
    ACTIVATION_FAST_GELU,       //!< FAST_GELU激活类型
    ACTIVATION_SWISH,           //!< SWISH激活类型
    ACTIVATION_LOG,             //!< LOG激活类型
    ACTIVATION_SWIGLU_FORWARD,  //!< SWIGLU_FORWARD激活类型
    ACTIVATION_SWIGLU_BACKWARD, //!< SWIGLU_BACKWARD激活类型
    ACTIVATION_SIGMOID,         //!< SIGMOID激活类型
    ACTIVATION_MAX,             //!< 枚举最大值
};

//!
//! \enum CommMode
//!
//! \brief 通信算子支持的通信模式.
//!
enum CommMode : int {
    COMM_UNDEFINED = -1, //!< 未定义
    COMM_MULTI_PROCESS,  //!< 指定多进程通信
    COMM_MULTI_THREAD,   //!< 指定多线程通信
};

//!
//! \brief 激活函数。
//!
struct ActivationParam {
    //! \enum GeLUMode
    //! \brief GeLU激活函数可选的计算模式
    enum GeLUMode : int {
        TANH_MODE = 0,  //!< 默认值，使用tanh估算
        NONE_MODE,      //!< 原GeLU计算公式
    };
    //! 激活函数类型，ActivationType类型枚举值.
    ActivationType activationType = ACTIVATION_UNDEFINED;
    //! SWISH激活函数的参数.
    float scale = 1.0f;
    //! SWIGLU激活函数的参数.
    int32_t dim = -1;
    //! GeLU模式选择参数
    GeLUMode geluMode = TANH_MODE;
};

//!
//! \brief InTensor根据指定参数，生成一个数据重新排布过的OutTensor.
//!
//! \warning 输出y基于输入x的总偏移量要求小于输入x的大小.
//!
struct AsStridedParam {
    //!
    //! \brief OutTensor的shape.
    //!
    //! \warning size的长度要求小于或等于8且各元素要求大于0.
    //!
    SVector<int64_t> size;
    //!
    //! \brief 用于从InTensor推导OutTensor的各维度的步长.
    //!
    //! \warning stride的长度要求与size一致，各元素要求大于或等于0.
    //!
    SVector<int64_t> stride;
    //!
    //! \brief OutTensor内存相对于InTensor内存的偏移，作为常数使用.
    //!
    //! \warning offset的长度要求为1且元素要求大于或等于0.
    //!
    SVector<int64_t> offset;
};

//!
//! \brief 后处理累积和计算.
//!
struct CumsumParam {
    //!
    //! \brief 指定axis轴(维度)上计算累加和，只能包含一个轴索引.
    //!
    //! \warning axes的值必须小于输入x的维度数。
    //!
    SVector<int64_t> axes;
    //!
    //! \brief 在某一个轴上的累加结果从第几个元素开始，默认为false.
    //!
    //! \note true：从第一个元素开始（暂不支持） false：从第0个元素开始.
    //!
    bool exclusive = false;
    //!
    //! \brief 正向累加或逆向累加，默认为false.
    //!
    //! \note true：输出正向累加（暂不支持） false：输出逆向累加.
    //!
    bool reverse = false;
};

//!
//! \brief 从输入张量中根据索引收集切片，并将这些切片组合成一个新的张量.
//!
struct GatherParam {
    //!
    //! \brief 指定要收集切片的轴。默认值为0.
    //!
    //! \warning 该参数必须大于或等于0
    //!
    int64_t axis = 0;
    //!
    //! \brief  允许从一个batch的每个元素中收集不同的项目，默认值为0.
    //!
    //! \warning 该参数必须大于或等于0,且小于或等于axis.
    //!
    int64_t batchDims = 0;
};

//!
//! \brief 采样功能。对最后一个轴进行采样，随机抽取numSamples个值，输出下标。
//!
//! \warning 用户需确保对最后一个轴进行归一化操作。
//!
struct MultinomialParam {
    //!
    //! \brief 随机采样数.
    //!
    //! \warning 小于等于输入张量对应的维度大小，最大为64。
    //!
    uint32_t numSamples = 1;
    //! \brief 随机数种子.
    uint32_t randSeed = 0;
};

//!
//! \brief 对输入张量指定维度等分切成多个张量。
//!
struct SplitParam {
    //!
    //! \brief 指定切分的维度索引
    //!
    //! splitDim须位于输入张量x的维度范围内，即如果x的维度为xDim，则splitDim的取值范围为[-xDim, xDim - 1]。
    //! 当splitDim为负数时，其含义是从最高维度开始访问，如splitDim = -1，x维度数为dimNum，则拆分维度为dimNum - 1。
    //!
    int32_t splitDim = 0;
    //!
    //! \brief 等分次数,当前支持2或3.
    //!
    //! \warning 输入张量x的维度须能够被splitNum整除,且当splitNum = 3时输入x要求是float16或者bfloat16数据类型。
    //!
    int32_t splitNum = 2;
};

//!
//! \brief 将两个输入张量在指定维度拼接成一个输出张量
//!
struct ConcatParam {
    //!
    //! \brief 指定拼接的维度索引
    //!
    //! 当concatDim为负数时，其含义是从最高维度开始访问，如concatDim = -1，输入张量维度数为dimNum，则拼接维度为dimNum - 1。
    //!
    //! \warning 输入x和y的维数要求一致。输入x或y的维度大小，除了concatDim维外，其他维度要求相同。仅Atlas 800I A2硬件支持bfloat16。
    //!
    int concatDim = 0;
};

//!
//! \brief 从输入张量某个起始位置中提取指定大小的切片
//!
//!
struct SliceParam {
    //!
    //! \brief 每个维度切片的起始位置
    //!
    //! 当offsets[i]为负数时，其含义是第i维最高维度开始访问，如offsets= -1，输入x的维度为dimNum，则对应维度切片的起始位置为dimNum - 1。
    //!
    //! \warning 当offsets元素x小于0时，该元素对应的维度大小为dimNum，要求dimNum与x之和大于等于0。
    //!
    SVector<int64_t> offsets;
    //!
    //! \brief 每个维度切片的大小
    //!
    //! 当size = -1时，表示切片的结束位置是对应维度最后一个位置。如果对应维度大小为dimNum，则结束位置为dimNum - 1。
    //!
    //! \warning size中元素要求大于等于-1。对应维度offsets，以及offsets + size须在x的对应维度的大小范围内。
    //!
    SVector<int64_t> size;
};

//!
//! \brief Softmax多分类激活函数，将多维（最大8维）Tensor数据在指定轴上映射到0到1之间，且非softmax轴数值之和为1。
//!
struct SoftmaxParam {
    //!
    //! \brief 指定轴（维度），axes可以支持多个轴上进行处理
    //!
    //! \warning axes不能为空，当指定多个轴时，多个轴之间必须连续且从小到大排列。
    //! \warning axes的元素要求大于或等于-1且小于输入x的维度
    //!
    SVector<int64_t> axes;
};

//!
//! \brief 改变输入Tensor的排列顺序，在多个维度上进行转置
//!
struct TransposeParam {
    //! 指示输入维度的重排结果, 需要保证输入正确，维度和输入x一致
    SVector<int32_t> perm;
};

//!
//! \struct ElewiseParam
//!
//! \brief 常用的逐元素数值计算集合
//!
//! ELEWISE_ADD、ELEWISE_MUL、ELEWISE_REALDIV、ELEWISE_SUB计算类型将会对输入进行广播后再进行指定操作。
//! 输入x、y对应维度的对应值要求相同或至少其中一个为1
//!
struct ElewiseParam {
    //!
    //! \enum ElewiseType
    //!
    //! \brief 计算类型
    //!
    enum ElewiseType : int {
        ELEWISE_UNDEFINED = 0,        //!< 默认值，未定义
        ELEWISE_CAST,                 //!< 数据类型转换
        ELEWISE_MULS,                 //!< 向量逐元素乘值
        ELEWISE_COS,                  //!< 逐元素计算余弦值
        ELEWISE_SIN,                  //!< 逐元素计算正弦值
        ELEWISE_NEG,                  //!< 逐元素取相反数
        ELEWISE_QUANT,                //!< 量化
        ELEWISE_LOGICAL_NOT,          //!< 逐元素逻辑非
        ELEWISE_ADD,                  //!< 逐元素相加
        ELEWISE_MUL,                  //!< 向量与向量逐元素相乘
        ELEWISE_REALDIV,              //!< 向量与向量逐元素相除
        ELEWISE_LOGICAL_AND,          //!< 逐元素逻辑与
        ELEWISE_LOGICAL_OR,           //!< 逐元素逻辑或
        ELEWISE_LESS,                 //!< 逐元素判断是否小于
        ELEWISE_GREATER,              //!< 逐元素判断是否大于
        ELEWISE_SUB,                  //!< 逐元素相减
        ELEWISE_EQUAL,                //!< 逐元素判断是否相等
        ELEWISE_QUANT_PER_CHANNEL,    //!< 每个通道量化
        ELEWISE_DEQUANT_PER_CHANNEL,  //!< 每个通道反量化
        ELEWISE_DYNAMIC_QUANT,        //!< 逐行动态量化
        ELEWISE_TANH,                 //!< 逐元素计算双曲正切值
    };

    //! 量化（非每通道）所需参数
    struct QuantParam {
        //! 量化的步长
        float inputScale = 1.0f;
        //! 动态量化的是否为非对称量化
        bool asymmetric = false; //!< false : symmetric，true : asymmetric
        //! 量化的偏移度
        int inputOffset = 0;
    };

    //! 向量乘值所需参数
    struct MulsParam {
        //! 向量乘的值
        float varAttr = 0.0f;
    };

    //! 计算方式
    ElewiseType elewiseType = ELEWISE_UNDEFINED;
    //! 量化参数
    QuantParam quantParam;
    //! 乘值参数
    MulsParam mulsParam;
    //! 指定数据类型转换输出的数据类型
    aclDataType outTensorType = ACL_DT_UNDEFINED;
};

//!
//! \struct KvCacheParam
//!
//! \brief KVCache处理。
//!
struct KvCacheParam {};

//!
//! \struct GatingParam
//!
//! \brief 主要功能为将token和专家的映射关系反转为专家与token的映射关系。算子输入为MoE模型每个token选中专家的索引，算子输出为MoE模型每个专家对应的token的索引。
//!
struct GatingParam {
    //! \brief 每个token选中的专家数。取值大于0。
    int32_t topkExpertNum = 0;
    //! \brief 专家总数。取值范围为[0, 127]。
    int32_t cumSumNum = 0;
};

//!
//! \brief 遍历每个key和value，将key和value(num_heads, head_size)按照slotmapping填入key_cache/value_cache指定位置
//!
struct ReshapeAndCacheParam {
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0,  //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD          //!< 压缩key_cache, value_cahe的kvHead维度
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
};

//!
//! \struct LayerNormParam
//!
//! \brief LayerNorm归一化处理。当前支持三种：NORM、PRENORM、POSTNORM。
//!
//! \warning beginNormAxis维度小于等于输入x的维度。
//! 所有输入输出Tensor的最后一维大小相等。
//!
struct LayerNormParam {
    //!
    //! \enum LayerNormType
    //!
    //! \brief 归一化类型：NORM、PRENORM、POSTNORM。
    //!
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0,  //!< 默认值，未定义
        LAYER_NORM_NORM,           //!< norm
        LAYER_NORM_PRENORM,        //!< prenorm
        LAYER_NORM_POSTNORM,       //!< postnorm
        LAYER_NORM_MAX,
    };
    //!
    //! \brief NORM参数。
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 归一化的维度，默认值为0，从第几维开始norm，同时决定输入gamma和beta维度。
        int32_t beginNormAxis = 0;
        //! \brief 归一化的维度，默认值为0，决定从第几维开始把后面的维度按轴合并。
        int32_t beginParamsAxis = 0;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
    };
    //!
    //! \brief PRENORM参数
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNDEINFED。
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        size_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
    };
    //!
    //! \brief POSTNORM参数。
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        size_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
    };
    //! \brief layerType
    LayerNormType layerType = LAYER_NORM_UNDEFINED;
    //! \brief normParam
    NormParam normParam;
    //! \brief preNormParam
    PreNormParam preNormParam;
    //! \brief postNormParam
    PostNormParam postNormParam;
};

//!
//! \struct RmsNormParam
//!
//! \brief RMS归一化处理。
//!
//! \warning 所有输入输出Tensor的最后一维大小相等。
//!
struct RmsNormParam {
    //!
    //! \brief RmsNormType
    //!
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0,  //!< 默认值，未定义
        RMS_NORM_NORM,           //!< NORM参数。
        RMS_NORM_PRENORM,        //!< PRENORM参数。
        RMS_NORM_POSTNORM,       //!< POSTNORM参数
    };
    //!
    //! \brief PrecisionMode
    //!
    enum PrecisionMode : int {
        HIGH_PRECISION_MODE = 0,  //!< 中间计算使用fp32类型
        HIGH_PERFORMANCE_MODE,    //!< 中间计算使用fp16类型
    };
    //!
    //! \brief ModelType
    //!
    enum ModelType : int {
        LLAMA_MODEL = 0,  //!< 默认值，使用Llama rmsnorm的公式
        GEMMA_MODEL,    //!< 使用Gemma rmsnorm的公式
    };
    //!
    //! \brief NormParam
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED, QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief Epsilon，默认为1e-5，暂时不使用。
        double layerNormEps = 1e-5;
        //! \brief 默认为False，设置为true时会使用训练的rmsnormforward算子。仅在Atlas 800I A2推理产品上支持该设置。
        //!  不支持和“precisionMode”，“modelType”同时设置。量化场景下不支持使用“rstd”。
        bool rstd = false;
        //! \brief 默认为HIGH_PRECISION_MODE。
        //! 支持参数如下：
        //! HIGH_PRECISION_MODE：默认值，中间计算使用fp32类型
        //! HIGH_PERFORMANCE_MODE： 中间计算使用fp16类型
        //! 不支持和“rstd”，“modelType”同时设置。
        //! 量化场景下不支持使用“precisionMode”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        //! \brief 默认为LLAMA_MODEL，设置为GEMMA_MODEL时使用gemma模型的rmsnorm计算公式。
        //! 支持参数如下：
        //! LLAMA_MODEL：默认值， Llama的rms norm计算公式。
        //! GEMMA_MODEL：Gemma的rms norm计算公式。
        //! 不支持和“rstd”，“precisionMode”同时启用。
        //! 量化场景下不支持使用“modelType”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        ModelType modelType = LLAMA_MODEL;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
    };
    //!
    //! \brief PreNormParam
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。量化场景下不支持使用“hasBias”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        bool hasBias = false;
    };
    //!
    //! \brief PostNormParam
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNDEINFED。
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。
        bool hasBias = false;
    };
    //! \brief 归一化类型，参数如下：
    //! RMS_NORM_UNDEFINED：默认值，未定义。
    //! RMS_NORM_NORM：NORM参数。
    //! RMS_NORM_PRENORM：PRENORM参数。
    //! RMS_NORM_POSTNORM：POSTNORM参数。
    RmsNormType layerType = RMS_NORM_UNDEFINED;
    //! \brief NORM参数。
    NormParam normParam;
    //! \brief PRENORM参数。
    PreNormParam preNormParam;
    //! \brief POSTNORM参数。
    PostNormParam postNormParam;
};

//!
//! \struct FillParam
//!
//! \brief 将指定位置设置为value值或者生成一个指定Shape的Tensor并填充为value。
//!
//! \warning 输入x不可以被broadcast。输入mask的元素只能是0或者1，且可以被broadcast。
//!
struct FillParam {
    //! \brief 是否Masked Fill。
    bool withMask = true;
    //! \brief 填充的元素，value是一个只含有一个元素的SVector。
    SVector<float> value;
    //! \brief withMask = false时，表示输出Tensor的Shape。
    SVector<int64_t> outDim;
};

//!
//! \struct AllGatherParam
//!
//! \brief 将多个通信卡上的数据按所属rank号的顺序在第一维进行聚合，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllGatherParam {
    //! \brief 每张卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl",Atlas 推理系列产品（配置Atlas 300I DUO）仅支持backend为"hccl"。
    //!
    //! 推理系列产品（配置Atlas 300I DUO）不支持bf16。
    //! 当backend为"lccl"时，且若机器拓扑为Atlas 800I A2单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

//!
//! \struct AllReduceParam
//!
//! \brief 将多个通信卡上的数据进行计算，支持相加、取最大、最小、相乘四种计算，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllReduceParam {
    //! \brief 每张卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 通信计算类型，支持"sum","prod","max"和"min".
    std::string allReduceType = "sum";
    //!
    //! \brief 通信计算类型，仅支持"hccl"和"lccl".推理系列产品（配置Atlas 300I DUO）仅支持backend为"hccl"。
    //!
    //! backend为"hccl"时，支持"sum","prod","max"和"min"; backend为"lccl"时，支持"sum","max"和"min".
    //! 当backend为"hccl"时，allReduceType为"prod"时，不支持数据类型为int16和bf16。
    //! 当backend为"hccl"时，推理系列产品（配置Atlas 300I DUO）不支持int64,bf16,int16只有allReduceType为"sum"时支持
    //! 当backend为"lccl"时，不支持数据类型int64，且若机器拓扑为Atlas 800I A2单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

//!
//! \struct BroadcastParam
//!
//! \brief 将通信主卡上的数据广播到其他每张卡上, 该算子不支持推理系列产品（配置Atlas 300I DUO）。
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct BroadcastParam {
    //! \brief 每张卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief HCCL通信域指针.
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl"。
    std::string backend = "hccl";
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

//!
//! \struct LinearParam
//!
//! \brief 将A、B两个矩阵进行矩阵乘运算，同时可以选择对矩阵乘的运算结果添加偏置或进行反量化操作。
//!
//! 算子本质上是接收x和weight两个输入tensor作为A矩阵和B矩阵进行矩阵乘运算，可通过参数transposeA与transposeB控制做矩阵乘前是否需要对A矩阵和B矩阵进行行列转置，
//! 根据参数转置后的A矩阵和B矩阵需满足矩阵乘维度关系，即A矩阵最后一维与B矩阵第0维相等。该算子分为浮点和量化两类，可通过输出数据类型进行选择。
//!
//! \warning 在Atlas 推理系列产品（配置Atlas 300I DUO）中，不支持BF16数据类型的计算，即输入和输出张量的数据类型均不支持BF16。
//!
struct LinearParam {
    //! \brief 是否转置A矩阵，默认不转置。
    //!
    //! 当输入x的维度为3时，transposeA必须为false。
    //! 在Atlas 推理系列产品（配置Atlas 300I DUO）中，量化情况下，transposeA必须为false。
    bool transposeA = false;
    //! \brief 是否转置B矩阵，默认转置。
    //!
    //! 在Atlas 推理系列产品（配置Atlas 300I DUO）中，量化情况下，transposeB必须为true。
    bool transposeB = true;
    //! \brief 是否叠加偏置。
    //!
    //! 在Atlas 推理系列产品（配置Atlas 300I DUO）中，量化情况下，hasBias必须为true。
    bool hasBias = true;
    //! \brief 输出数据类型.
    //!
    //! 若为浮点linear，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致；
    //! 若为量化linear，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型，
    //! 目前仅支持ACL_FLOAT16/ACL_BF16，在Atlas 推理系列产品（配置Atlas 300I DUO）中，不支持ACL_BF16。
    aclDataType outDataType = ACL_DT_UNDEFINED;
};

//!
//! \struct LinearParallelParam
//!
//! \brief 通信计算并行算子,该算子功能为linear和通信算子组合
//!
//! 通信和计算是并行处理，和串行相比存在大幅度性能提升.
//!
//! \see LinearParam,AllReduceParam,AllGatherParam
//!
struct LinearParallelParam {
    //!
    //! \enum ParallelType
    //!
    //! \brief 通信类型
    //!
    enum ParallelType : int {
        UNDEFINED = -1,            //!< 默认值
        LINEAR_ALL_REDUCE = 0,     //!< linear+AllReduce
        LINEAR_REDUCE_SCATTER = 1, //!< linear+reduce_scatter
        ALL_GATHER_LINEAR = 2,     //!< AllGather+linear
        PURE_LINEAR = 3,           //!< linear
        MAX = 4,                   //!< 枚举类型最大值
    };
    //!
    //! \enum QuantType
    //!
    //! \brief QuantType类型
    //!
    enum QuantType : int {
        QUANT_TYPE_UNDEFINED = -1,  //!< 默认值
        QUANT_TYPE_PER_TENSOR = 0,  //!< 对整个张量进行量化
        QUANT_TYPE_PER_CHANNEL = 1, //!< 对张量中每个channel分别进行量化
        QUANT_TYPE_PER_GROUP = 2,   //!< 将张量按quantGroupSize划分后，分别进行量化
        QUANT_TYPE_MAX = 3,         //!< 枚举类型最大值
    };
    //! \brief 权重是否需要转置，默认为true。
    bool transWeight = true;
    //! \brief 每张卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 是否叠加残差。配置为false时不叠加残差，为true时叠加残差。默认不叠加残差。
    bool hasResidual = false;
    //! \brief 通信后端指示。支持"hccl"，"lccl"，"lcoc"。
    std::string backend = "hccl";
    //! \brief HCCL通信域接口获取的地址指针，仅当"hcclComm"不为nullptr时可用。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景。
    std::string rankTableFile;
    //! \brief 权重并行类型。
    ParallelType type = LINEAR_ALL_REDUCE;
    //! \brief 是否返回中间结果，仅在使用ALL_GATHER_LINEAR时生效。
    bool keepIntermediate = false;
    //! \brief 量化类型。
    QuantType quantType = QUANT_TYPE_UNDEFINED;
    //! \brief 量化类型为QUANT_TYPE_PER_GROUP时生效。
    int32_t quantGroupSize = 0;
    //!
    //! 若为浮点linear，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致,
    //! 若为量化linear，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型,
    //! 如ACL_FLOAT16/ACL_BF16
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

//!
//! \struct LinearSparseParam
//!
//! \brief 稀疏量化linear
//!
//! 该算子实现功能与量化linear类似。不同点在于稀疏量化算子会使用压缩工具提前对weight输入进行压缩，
//! 以此提升算子性能。参数tilingK和tilingN由压缩算法决定，目前均只支持取值为8.
//! 目前该算子仅支持在Atlas 推理系列产品（配置Atlas 300I DUO）中进行运算。
//!
struct LinearSparseParam {
    //! \brief 是否转置A矩阵，默认不转置。当前仅支持transposeA = false。
    bool transposeA = false;
    //! \brief 是否转置B矩阵，默认转置。当前仅支持transposeB = true。
    bool transposeB = true;
    //! \brief 压缩参数，由外部压缩算法决定，默认为1，目前仅支持取值为8。
    uint32_t tilingK = 1;
    //! \brief 压缩参数，由外部压缩算法决定，默认为1，目前仅支持取值为8。
    uint32_t tilingN = 1;
};

//!
//! \struct FfnParam
//!
//! \brief 暂不支持
//!
struct FfnParam {
    //! \brief 暂不支持
    bool firstTransposeA = false;
    //! \brief 暂不支持
    bool firstTransposeB = false;
    //! \brief 暂不支持
    bool firstHasBias = true;
    //! \brief 暂不支持
    ActivationType activationType = ACTIVATION_FAST_GELU;
    //! \brief 暂不支持
    bool secondTransposeA = false;
    //! \brief 暂不支持
    bool secondTransposeB = false;
    //! \brief 暂不支持
    bool secondHasBias = true;
};

//!
//! \struct FfnQuantParam
//!
//! \brief 暂不支持
//!
struct FfnQuantParam {
    //! \brief 暂不支持
    LinearParam firstLinearParam;
    //! \brief 暂不支持
    ActivationType activationFuncType = ACTIVATION_FAST_GELU;
    //! \brief 暂不支持
    LinearParam secondLinearParam;
    //! \brief 暂不支持
    float inputScale = 1;
    //! \brief 暂不支持
    int inputOffset = 0;
};

//!
//! \brief 旋转位置编码。hiddenSizeQ必须是hiddenSizeK的整数倍且满足hiddenSizeQ = headDim * headNum。
//!
struct RopeParam {
    //! \brief rope，旋转系数，对半旋转是2，支持配置2、4或headDim / 2。
    int32_t rotaryCoeff = 4;
    //! \brief 训练用参数，支持配置0或1
    int32_t cosFormat = 0;
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const RopeParam &left, const RopeParam &right)
{
    return left.rotaryCoeff == right.rotaryCoeff && left.cosFormat == right.cosFormat;
}

//!
//! \brief KVCache+KVCache+Muls+FlashAttention.
//!
struct SelfAttentionParam {
    //!
    //! \enum CalcType
    //!
    //! \brief 计算类型
    //!
    enum CalcType : int {
        UNDEFINED = 0, //!< decoder&encoder for flashAttention
        ENCODER,       //!< encoder for flashAttention
        DECODER,       //!< decoder for flashAttention
        PA_ENCODER     //!< encoder for pagedAttention
    };
    //!
    //! \enum KernelType
    //!
    //! \brief 算子内核精度类型
    //!
    enum KernelType : int {
        KERNELTYPE_DEFAULT = 0,   //!< i:fp16, bmm:fp16, o:fp16
        KERNELTYPE_HIGH_PRECISION //!< i:fp16, bmm:fp32, o:fp16
    };
    //!
    //! \enum ClampType
    //!
    //! \brief clamp类型
    //!
    enum ClampType : int {
        CLAMP_TYPE_UNDEFINED = 0, //!< 不做clamp
        CLAMP_TYPE_MIN_MAX        //!< 做clamp，同时指定最大最小值
    };
    //!
    //! \enum MaskType
    //!
    //! \brief mask类型
    //!
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0,              //!< 默认值，全0mask
        MASK_TYPE_NORM,                       //!< 倒三角mask
        MASK_TYPE_ALIBI,                      //!< alibi mask
        MASK_TYPE_NORM_COMPRESS,              //!< 倒三角压缩mask
        MASK_TYPE_ALIBI_COMPRESS,             //!< alibi压缩mask
        MASK_TYPE_ALIBI_COMPRESS_SQRT,        //!< alibi压缩开平方mask
        MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN   //!< alibi压缩mask左对齐,只支持Atlas 800I A2
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置,不支持calcType为PA_ENCODER
    //!
    enum KvCacheCfg :int {
        K_CACHE_V_CACHE = 0, //!< 默认值,进行kvcache处理
        K_BYPASS_V_BYPASS,   //!< 直接传入kvcache
    };
    //! query头大小, 需大于或等于0
    int32_t headNum = 0;
    //! kv头数量, 该值需要用户根据使用的模型实际情况传入
    //! kvHeadNum = 0时，keyCache的k_head_num，valueCache的v_head_num与query的num_heads一致，均为num_heads的数值
    //! kvHeadNum != 0时，keyCache的k_head_num， valueCache的v_head_num与kvHeadNum值相同
    int32_t kvHeadNum = 0;
    //! query缩放系数
    float qScale = 1;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //! 是否开启倒三角优化, 只有mask为倒三角的时候才能开启优化
    uint32_t isTriuMask = 0;
    //! 计算类型
    CalcType calcType = UNDEFINED;
    //! 内核精度类型
    KernelType kernelType = KERNELTYPE_DEFAULT;
    //! clamp类型
    ClampType clampType = CLAMP_TYPE_UNDEFINED;
    //! clamp功能最小值
    float clampMin = 0;
    //! clamp功能最大值
    float clampMax = 0;
    //! mask类型
    MaskType maskType = MASK_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvcacheCfg = K_CACHE_V_CACHE;
};

//!
//! \brief PagedAttention.
//!
//! 一个Q有多个token，一个token对应多个KV的token，以token0为例，block_table代表其对应的KV的block_id，-1代表截止，
//! 所以第二行和第四行为其目标block，context_lens则表示KV有多少个token，则代表仅有block_id为(3,4,5,9,10)是需要与Q进行计算的。
//!
struct PagedAttentionParam {
    //! query 头大小
    int32_t headNum = 0;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1.0;
    //! kv头数量
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief The type values of MaskType.
    //!
    enum MaskType : int {
        UNDEFINED = 0,          //!< 默认值，全0的mask
        MASK_TYPE_NORM,         //!< 倒三角mask
        MASK_TYPE_ALIBI,        //!< alibi mask
        MASK_TYPE_SPEC          //!< 并行解码mask
    };
    //! mask类型
    MaskType maskType = UNDEFINED;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //!
    //! \enum QuantType
    //!
    //! \brief quant类型
    //!
    enum QuantType : int {
        TYPE_QUANT_UNDEFINED = 0, //!< 默认值，不与量化融合
        TYPE_DEQUANT_FUSION       //!< 与反量化融合, 只支持Atlas 800I A2
    };
    //! 量化类型
    QuantType quantType = TYPE_QUANT_UNDEFINED;
    //! 开启量化功能后是否使用offset
    bool hasQuantOffset = false;
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0,  //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD          //!< 压缩key_cache, value_cahe的kvHead维度, 只支持Atlas 800I A2
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //!
    //! \enum CalcType
    //!
    //! \brief The type values of CalcType.
    //!
    enum CalcType : int {
        CALC_TYPE_UNDEFINED = 0,  //!< 默认值，不开启并行解码
        CALC_TYPE_SPEC            //!< 并行解码功能
    };
    //! 计算类型
    CalcType calcType = CALC_TYPE_UNDEFINED;
};

//!
//! \brief 数据格式转换处理。
//!
//! 使用的NZ的dims约定表示方式：{b, n1, m1m0, n0}，对应的ND的dims是{b, m, n}，
//! 其中：b表示batch，如果batch为1，该维度为1，不可省略。如果batch有多个，该维度为所有batch维度合轴的结果。
//! m0/n0表示对齐位，float16时，n0与m0都为16, int8时，n0为32，m0为16，m1m0表示原始ND的m维度经过对齐位向上对齐，
//! n1表示原始ND的n维度经过对齐位向上对齐后，除以n0的商。例如原始ND的dims为{8, 100, 30}，则其对应的NZ的dims为{8, 2, 112, 16}。
//!
//! \warning outCrops的长度要求是2，其值须满足以下要求：
//! - 如果m0m1落在区间(k1 × 16, (k1 + 1) × 16]（其中k1为正整数）内，那么该区间即为outCrops[0]的取值范围要求。
//! - 如果n0*n1落在区间(k2 × 16, (k2 + 1) × 16]（其中k2为正整数）内，那么该区间即为outCrops[1]的取值范围要求。
//!
struct TransdataParam {
    //!
    //! \enum TransdataType
    //!
    //! \brief TransdataType类型值
    //!
    enum TransdataType : int {
        UNDEFINED = 0,    //!< 默认
        FRACTAL_NZ_TO_ND, //!< FRACTAL_NZ转ND
        ND_TO_FRACTAL_NZ  //!< ND转FRACTAL_NZ
    };
    //! \brief 数据格式转换类型，支持FRACTAL_NZ和ND互相转换。
    TransdataType transdataType = UNDEFINED;
    //! \brief 仅当FRACTAL_NZ转ND时使用，表示原ND数据格式Shape的最后两维。
    SVector<int64_t> outCrops = { 0, 0 };
};

//!
//! \brief 三目运算。
//!
//! 输入张量为cond,x,y, 输出张量 z = cond ? x : y;
//! 输入cond的元素只能是0或者1
//! 输出z的维度为输入x与y广播后的结果。要求cond, x, y必须是可广播的。
//!
struct WhereParam {};

//!
//! \brief 将输入Tensor的Shape，按指定轴扩展指定的倍数。
//!
//! \warning 输出y的维度和multiples维度一致，每个维度大小为输入x广播到multiples维度后和multiples对应维度的乘积。
//!
struct RepeatParam {
    //!
    //! \brief 每一维度上扩展的倍数。
    //!
    //! \warning
    //! - 支持在不超过两个维度上进行扩展
    //! - multiples的维度小于等于8且需大于或等于输入x的维度，每一个元素要求大于0。
    //!
    SVector<int64_t> multiples;
};

//!
//! \struct SetValueParam
//!
//! \brief 将输入源张量中的内容拷贝到输入目标张量指定位置中.
//!
//! 该拷贝为原地拷贝，最终结果修改在输入目标张量中.<br>
//! 输入目标张量 dst: [a,b,c], 输入源张量src: [d,e,f].
//! dst[starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]] = src.<br>
//! 其中 ends[0]-starts[0]需为src第0维的维度大小,ends[1]-starts[1]需为为src第1维的维度大小,ends[2]-starts[2]需为src第2维的维度大小。
//!
//! \warning 输入src和输入dst的维数须相同.<br>
//! 输入src的各维度大小要求小于或等于输入dst对应维度大小.<br>
//! 输入src和输入dst的各维度要求有一个或两个维度不相同，且需要满足：
//!   - 如果有一个维度不相同，则这个维度不能是最高维（第0维）。
//!   - 如果有两个维度不相同，则其中一个不同的维度必须是最高维（第0维）。
//
struct SetValueParam {
    //! \brief 每一维拷贝起始位置
    SVector<int64_t> starts;
    //! \brief 每一维拷贝结束位置后一个位置，拷贝到该位置前一个位置为止
    SVector<int64_t> ends;
    //! \brief 每一维拷贝步长，当前仅支持strides为全1.
    SVector<int64_t> strides;
};

//!
//! \brief 在指定维度上求和、取最大值或最小值，并消除这个维度。
//!
struct ReduceParam {
    //!
    //! \enum ReduceType
    //!
    //! \brief ReduceType支持的值
    //!
    enum ReduceType {
        REDUCE_UNDEFINED = 0,  //!< 未定义。
        REDUCE_MAX,            //!< 求最大值。
        REDUCE_MIN,            //!< 求最小值。
        REDUCE_SUM,            //!< 求和。
    };
    //! \brief reduceType
    ReduceType reduceType = REDUCE_UNDEFINED;
    //!
    //! \brief 指定轴（维度）。
    //!
    //! \warning axis不能为空且长度要求小于等于输入x的维度。<br>
    //! axis可以支持多个轴上进行处理，各元素要求小于x的维度且大于等于0
    //!
    SVector<int64_t> axis;
};

//!
//! \brief 依据给定的词表概率以及top-p，设置随机种子及top-k保留词数，选择最合适的词及对应概率作为输出。
//!  支持btach级别随机种子、top-k取样，支持exponential取样
//! \warning probs必须是两维张量。
//!
struct TopkToppSamplingParam {
    //! \brief 取样处理类型
    enum TopkToppSamplingType {
        SAMPLING_UNDEFINED = -1, //!< 未定义
        SINGLE_TOPK_SAMPLING,  //!< 非batch级别随机种子、Topk的取样
        BATCH_TOPK_MULTINOMIAL_SAMPLING,    //!< batch级别随机种子、Topk的multinomial取样
        BATCH_TOPK_EXPONENTIAL_SAMPLING,    //!< batch级别随机种子、Topk的exponential取样
        SAMPLING_MAX,   //!< 枚举最大值
    };
    //! \brief 采样类型，默认为非batch级别随机种子、Topk的取样
    TopkToppSamplingType topkToppSamplingType = SINGLE_TOPK_SAMPLING;
    //! \brief 当 topkToppSamplingType为BATCH_TOPK_MULTINOMIAL_SAMPLING时使用
    //! \brief 每个batch下top-p阶段随机抽样使用的随机数种子。
    //! \brief 维度与batch大小一致。
    std::vector<uint32_t> randSeeds;
    //! \brief 当 topkToppSamplingType为SINGLE_TOPK_SAMPLING时使用
    //! \brief top-p阶段随机抽样使用的随机数种子。
    uint32_t randSeed = 0;
    //! \brief 当 topkToppSamplingType为SINGLE_TOPK_SAMPLING时使用
    //! \brief top-k阶段保留的词的个数,需要小于词表的词数。
    //! \brief top-k必须大于0且小于或等于输入probs最后一维的大小。
    uint32_t topk = 100;
};


//!
//! \struct PadParam
//!
//! \brief 对于输入input_ids，取出每个batch最后一个有效token的embedding向量
//!
struct PadParam {};

//!
//! \struct UnpadParam
//!
//! \brief 对于输入input_ids，把所有有效的token拼接在一起，并在最后补0
//!
struct UnpadParam {};

//!
//! \struct SortParam
//!
//! \brief 后处理计算功能。实现输入tensor在最后一维上降序排列，并保留最大的num个元素，输出排序后的tensor及各元素对应的索引。
//!
struct SortParam {
    //!
    //! \brief 排序后保留的最大的元素的数量。
    //!
    //! \warning num是一个仅含有一个值的SVector，该值需大于0且小于等于输入x最后一维的大小。
    //!
    SVector<int32_t> num;
};

//!
//! \struct NonzeroParam
//!
//! \brief 输出非零值索引。
//!
//! \warning 仅在Atlas 800I A2硬件上支持
//!
struct NonzeroParam {};

//!
//! \struct OnehotParam
//!
//! \brief onehot编码。
//!
struct OnehotParam {
    //! \brief depth所在下标。可为负数。
    int64_t axis = 0;
    //! \brief 类别数。
    int64_t depth = 0;
};

//!
//! \struct IndexAddParam
//!
//! \brief 固定维度的指定下标加上某个特定值。
//!
struct IndexAddParam {
    //!
    //! \enum IndexType
    //!
    //! \brief 指定下标需要执行的操作类型。
    //!
    enum IndexType {
        INDEX_UNDEFINED = 0,  //!< 默认值
        INDEX_ADD,            //!< 加
    };
    //! \brief 指定下标需要执行的操作类型。
    IndexType indexType = INDEX_UNDEFINED;
    //! \brief 输入Tensor需加上updates更新值的轴。可为负数。值小于var的维度数。
    int64_t axis = 0;
};

//!
//! \struct SendParam
//!
//! \brief 将当前通信卡的输入发送至指定通信卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize, 0 ≤ destRank < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct SendParam {
    //! \brief 每张卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信域内数据接收端的rank编号.
    uint32_t destRank = 1;
    //! \brief 通信后端指示，仅支持"hccl".
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

//!
//! \struct RecvParam
//!
//! \brief 从当前通信卡接收来自指定通信卡的数据.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize, 0 ≤ srcRank < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct RecvParam {
    //! \brief 每张卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信域内数据发送端的rank编号.
    uint32_t srcRank = 1;
    //! \brief 通信后端指示，仅支持"hccl".
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
};

}  // namespace infer
}  // namespace atb
#endif
