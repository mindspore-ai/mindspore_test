mindspore.mint.distributed.reduce
=====================================

.. py:function:: mindspore.mint.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)

    规约指定通信组中的张量，并将规约结果发送到目标为dst的进程(全局的进程编号)中，返回发送到目标进程的张量。

    .. note::
        - 只有目标为dst的进程(全局的进程编号)才会收到规约操作后的输出。
        - 当前支持PyNative模式，不支持Graph模式。
        - 其他进程需传入张量，该张量没有数学意义。

    参数：
        - **tensor** (Tensor) - 输入和输出规约的Tensor，输出会直接修改输入。
        - **dst** (int) - 指定接收输出的目标进程编号，只有该进程会接收规约操作后的输出结果。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor， `op` 和 `group` 不是字符串， `async_op` 不是bool， `op` 值非法。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
