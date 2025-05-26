mindspore.ops.matmul_reduce_scatter
===================================

.. py:function:: mindspore.ops.matmul_reduce_scatter(input, x2, group, world_size, *, reduce_op='sum', bias=None,\
                                                     comm_turn=0, trans_input=False, trans_x2=False) -> Tensor

    TP 切分场景下，实现 matmul 和 reducescatter 的融合，融合算子内部实现通信和计算流水并行。

    .. math::
        output = reducescatter(input@x2)

    .. warning::
        这是一个实验性 API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - matmul 的左矩阵，dtype 支持 float16、bfloat16，shape 支持二维，数据格式支持 ND。
        - **x2** (Tensor) - matmul 的右矩阵，dtype 需要和 ``input`` 一致，shape 支持二维，数据格式支持 ND。
        - **group** (str) - 通信组名称，可以由 ``create_group`` 方法创建，或者使用默认组
          ``mindspore.communication.GlobalComm.WORLD_COMM_GROUP`` 。
        - **world_size** (int) - 通信组的总进程数，要求与实际运行的卡数一致，支持 ``2`` 、 ``4`` 、 ``8`` 。

    关键字参数：
        - **reduce_op** (str, 可选) - reduce 操作类型。当前仅支持 ``'sum'`` 。默认 ``'sum'`` 。
        - **bias** (Tensor, 可选) - 当前仅支持 ``None`` 。默认 ``None`` 。
        - **comm_turn** (int, 可选) - 表示进程间通信切分粒度。当前仅支持 ``0`` 。默认 ``0`` 。
        - **trans_input** (bool, 可选) - 表示 ``input`` 是否转置。当前仅支持 ``False`` 。默认 ``False`` 。
        - **trans_x2** (bool, 可选) - 表示 ``x2`` 是否转置。默认 ``False`` 。

    返回：
        - **output** (Tensor) - matmul 和 reducescatter 融合计算的结果。

    .. note::
        - 使用该接口时，请确保驱动固件包和 CANN 包都为配套的 8.0.RC2 版本或者配套的更高版本，否则将会引发报错，比如 BUS ERROR 等。
        - ``input`` 的 shape 为 (m, k)， ``x2`` 的 shape 为 (k, n)，要求 k 相等，且 k 的取值范围为 [256, 65535)，要求 m 是 ``world_size`` 的\
          整数倍。 ``output`` 的 shape 为 (m * world_size, n)。
        - 一个模型中的通算融合算子仅支持相同通信组。

    异常：
        - **TypeError** - 参数的类型不对。
        - **RuntimeError** - ``input`` 或 ``x2`` 的 dtype 不是 float16 或 bfloat16。
        - **RuntimeError** - ``input`` 和 ``x2`` 的 dtype 不一致。
        - **RuntimeError** - ``input`` 或 ``x2`` 的 shape 不是二维。
        - **RuntimeError** - ``input`` shape 和 ``x2`` shape 的 k 不相等。
        - **RuntimeError** - k 小于 ``256`` 或大于等于 ``65535`` 。
        - **RuntimeError** - ``bias`` 不是 ``None`` 。
        - **RuntimeError** - ``group`` 不存在。
        - **RuntimeError** - ``world_size`` 与实际运行的卡数不一致。
        - **RuntimeError** - ``world_size`` 不等于 ``2`` 、 ``4`` 、 ``8`` 。
        - **RuntimeError** - ``reduce_op`` 不是 ``'sum'`` 。
        - **RuntimeError** - ``trans_input`` 为 ``True`` 。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在 2 卡环境下运行。
