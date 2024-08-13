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
#ifndef ATB_GRAPH_OP_H
#define ATB_GRAPH_OP_H
#include <string>
#include <memory>
#include "atb/operation.h"

//!
//! \file graph_op_builder.h
//!
//! \brief 定义图算子的构建方法
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \class GraphOpBuilder.
//!
//! \brief 图算子创建类，主要用于简化图算子创建过程.
//!
//! GraphOpBuilder类会通过Operation的输入输出关系组建出算子的拓扑图，并最终完成GraphOp的创建，优化了之前手动定义tensor id的组图方式.
//!
class GraphOpBuilder {
public:
    GraphOpBuilder();
    virtual ~GraphOpBuilder();

    //!
    //! \brief 初始化图算子.
    //!
    //! 定义图算子的名称，输入输出及shape传导规则.
    //!
    //! \param opName 图算子的名称
    //!
    //! \param inferShapeFunc 图算子的shape传导规则
    //!
    //! \param inTensorNames 输入tensor名称
    //!
    //! \param outTensorNames 输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status Init(const std::string &opName, const InferShapeFunc &inferShapeFunc,
        const SVector<std::string> &inTensorNames, const SVector<std::string> &outTensorNames) = 0;

    //!
    //! \brief 改变输入tensor的shape.
    //!
    //! \param srcTensorName 输入tensor的名称
    //!
    //! \param reshapeFunc shape修改规则
    //!
    //! \param viewTensorName shape修改后的tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status Reshape(
        const std::string &srcTensorName, const ReshapeFunc &reshapeFunc, const std::string &viewTensorName) = 0;

    //!
    //! \brief 向图中添加算子.
    //!
    //! \param operation 要添加的算子
    //!
    //! \param inTensorNames 被添加算子的输入tensor名称
    //!
    //! \param outTensorNames 被添加算子的输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status AddOperation(Operation *operation, const SVector<std::string> &inTensorNames,
        const SVector<std::string> &outTensorNames) = 0;
    
    //!
    //! \brief 创建图算子.
    //!
    //! \return 返回被创建的图算子，失败返回空指针.
    //!
    virtual Operation *Build() = 0;
    
    //!
    //! \brief 创建并向图中添加算子.
    //!
    //! \param opParam 要添加的算子参数
    //!
    //! \param inTensorNames 被添加算子的输入tensor名称
    //!
    //! \param outTensorNames 被添加算子的输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    template <class OpParam>
    Status AddOperation(const OpParam &opParam, const SVector<std::string> &inTensorNames,
        const SVector<std::string> &outTensorNames)
    {
        Operation *operation = nullptr;
        Status st = CreateOperation(opParam, &operation);
        if (st != NO_ERROR) {
            return st;
        }

        return AddOperation(operation, inTensorNames, outTensorNames);
    }
};

//!
//! \brief 创建图算子构建器.
//!
//! \param builder 返回被创建的构建器
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateGraphOpBuilder(GraphOpBuilder **builder);

//!
//! \brief 销毁图算子构建器.
//!
//! \param builder 待销毁的构建器
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status DestroyGraphOpBuilder(GraphOpBuilder *builder);
}  // namespace atb
#endif