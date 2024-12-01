mindspore.mint.nn.ReplicationPad3d
==================================

.. py:class:: mindspore.mint.nn.ReplicationPad3d(padding)

    使用输入边界值对输入最后3维 `input` 进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入Tensor，4D或5D。shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，填充后的Tensor。

    异常：
        - **TypeError** - `padding` 不是一个integer或包含6个int的tuple或者list。
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `padding` 是tuple或list，且长度和Tensor的维度不匹配。
