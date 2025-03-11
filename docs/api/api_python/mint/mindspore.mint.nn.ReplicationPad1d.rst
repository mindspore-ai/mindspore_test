mindspore.mint.nn.ReplicationPad1d
==================================

.. py:class:: mindspore.mint.nn.ReplicationPad1d(padding)

    使用输入边界值，对输入的最后一维 `input` 进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入Tensor，维度为二维或三维。shape为 :math:`(C, W_{in})` 或 :math:`(N, C, W_{in})` 。

    输出：
        填充后的Tensor。

    异常：
        - **TypeError** - `padding` 不是一个int，或包含2个int的tuple或list。
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `padding` 是tuple或list，且长度和Tensor的维度不匹配。
