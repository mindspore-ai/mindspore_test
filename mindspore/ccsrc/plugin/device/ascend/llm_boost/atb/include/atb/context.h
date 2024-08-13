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
#ifndef ATB_CONTEXT_H
#define ATB_CONTEXT_H
#include <acl/acl.h>
#include "atb/types.h"

//!
//! \file context.h
//!
//! \brief 定义加速库上下文类
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \class Context.
//!
//! \brief 加速库上下文类，主要用于管理Operation运行所需要的全局资源.
//!
//! Context类会管理任务流队列比如Operation执行以及TilingCopy,管理tiling内存的申请与释放.
//!
class Context {
public:
    Context() = default;
    virtual ~Context() = default;
    
    //!
    //! \brief 将传入stream队列设置为当前执行队列.
    //!
    //! 将传入stream队列设置为当前执行队列,然后再去执行对应的Operation.
    //!
    //! \param stream 传入的stream队列
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetExecuteStream(aclrtStream stream) = 0;

    //!
    //! \brief 获取当前执行stream队列.
    //!
    //! \return 执行流队列
    //!
    virtual aclrtStream GetExecuteStream() const = 0;

    //!
    //! \brief 设置异步拷贝tiling信息功能.
    //!
    //! 设置异步拷贝tiling信息功能是否开启，如果是，则创建stream和event来进行tiling拷贝过程.
    //!
    //! \param enable 传入的标志，bool类型
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetAsyncTilingCopyStatus(bool enable) = 0;

    //!
    //! \brief 获取tiling拷贝状态.
    //!
    //! \return 如果获取成功，返回True.
    //!
    virtual bool GetAsyncTilingCopyStatus() const = 0;
};

//!
//! \brief 创建上下文.
//!
//! 在当前进程或线程中显式创建一个Context.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateContext(Context **context);

//!
//! \brief 销毁上下文.
//!
//! 销毁上下文中所有的资源.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status DestroyContext(Context *context);
}
#endif