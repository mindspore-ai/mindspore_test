mindspore.numpy.asfarray
=================================

.. py:function:: mindspore.numpy.asfarray(a, dtype=mstype.float32)

    类似于 ``asarray`` ，将输入转换为float tensor。如果 ``dtype`` 定义的是一个non-float格式，则该函数将返回float32格式的Tensor。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 可以转换为Tensor的任何形式的输入数据。 包含 ``int, float, bool, Tensor, list, tuple`` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定Tensor的数据类型，可以是 ``np.int32`` 或 ``int32`` 。如果 ``dtype`` 为 ``None`` ，则将从 ``a`` 推断出新Tensor的数据类型。默认值： ``mstype.float32`` 。

    返回：
        Tensor，具有指定的float数据类型。

    异常：
        - **TypeError** - 如果没有按照上述内容给定的数据类型输入参数。
        - **ValueError** - 如果输入 ``a`` 在不同的维度上具有不同的大小。