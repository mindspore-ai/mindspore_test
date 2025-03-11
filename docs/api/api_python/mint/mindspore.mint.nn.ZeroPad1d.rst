mindspore.mint.nn.ZeroPad1d
===========================

.. py:class:: mindspore.mint.nn.ZeroPad1d(padding)

    根据参数 `padding` ，对输入 `input` 最后1维填充零。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入Tensor。shape为 :math:`(N, *)`，其中 :math:`*` 表示任意维度。

    输出：
        Tensor，填充后的Tensor。

    异常：
        - **TypeError** - `padding` 既不是一个int，也不是包含2个int的tuple，或list。
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `padding` 含有负数。
        - **ValueError** - `padding` 是tuple或list，且长度和Tensor的维度不匹配。
