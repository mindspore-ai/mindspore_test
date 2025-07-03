mindspore.mint.nn.ZeroPad3d
===========================

.. py:class:: mindspore.mint.nn.ZeroPad3d(padding)

    根据参数 `padding` ，对输入 `input` 最后3维填充零。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(N, *)`，其中 :math:`*` 表示任意维度。

    输出：
        Tensor，填充后的Tensor。

    异常：
        - **TypeError** - `padding` 不是一个integer，或包含6个int的tuple，或list。
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `padding` 含有负数。
        - **ValueError** - `padding` 是tuple或list，且长度和Tensor的维度不匹配。
