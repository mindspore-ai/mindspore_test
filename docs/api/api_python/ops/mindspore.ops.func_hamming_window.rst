mindspore.ops.hamming_window
============================

.. py:function:: mindspore.ops.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None)

    汉明窗口函数。

    .. math::
        w[n]=\alpha − \beta \cos \left( \frac{2 \pi n}{N - 1} \right),

    其中， :math:`N` 是总的窗口长度 `window_length` ，n为小于 :math:`N` 的自然数 [0, 1, ..., N-1]。

    参数：
        - **window_length** (int) - 窗口的大小。
        - **periodic** (bool, 可选) - 如果为 ``True`` ，表示返回周期窗口。如果为 ``False`` ，表示返回对称窗口。默认 ``True`` 。
        - **alpha** (float, 可选) - 系数α。默认 ``0.54`` 。
        - **beta** (float, 可选) - 系数β。默认 ``0.46`` 。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        一维tensor。
