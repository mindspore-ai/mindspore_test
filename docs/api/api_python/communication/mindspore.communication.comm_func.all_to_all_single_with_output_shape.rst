mindspore.communication.comm_func.all_to_all_single_with_output_shape
======================================================================

.. py:function:: mindspore.communication.comm_func.all_to_all_single_with_output_shape(output_shape, tensor, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)

    根据用户输入的切分大小，把输入 `tensor` 切分后，发送到其他的设备上，并从其他设备接收切分块，然后合并到一个输出Tensor中。

    .. note::
        各个rank之间发送和接收的切分块大小需要互相匹配。
        仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **output_shape** (Union(Tensor, Tuple(int))) - 表示接收Tensor的形状。
        - **tensor** (Tensor) - 要发送到远端设备的Tensor。
        - **output_split_sizes** (Union(Tuple(int), List(int))) - 接收Tensor在0维的切分大小列表。默认值：None，表示均匀切分。
        - **input_split_sizes** (Union(Tuple(int), List(int))) - 发送Tensor在0维的切分大小列表。默认值：None，表示均匀切分。
        - **group** (str, 可选) - 通信执行所在的通信组。默认值：None。为None时，在Ascend上将使用为 ``hccl_world_group``，在GPU上使用 ``nccl_world_group``。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        Tuple(Tensor, CommHandle)，其中Tensor为从远端设备接收分块并合并的结果。
        如果从其他设备接收的Tensor为空，它将返回一个没有实际意义的、值为0的Tensor。
        若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - `tensor` 不是Tensor类型。
        - **TypeError** - `output_shape` 不是元组或者Tensor类型。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
