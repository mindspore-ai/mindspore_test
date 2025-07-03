mindspore.ops.hann_window
==========================

.. py:function:: mindspore.ops.hann_window(window_length, periodic=True, *, dtype=None)

    汉宁窗口函数。

    .. math::
        w(n) = \frac{1}{2} - \frac{1}{2} \cos\left(\frac{2\pi{n}}{M-1}\right),\qquad 0 \leq n \leq M-1

    参数：
        - **window_length** (int) - 窗口的大小。
        - **periodic** (bool, 可选) - 如果为 ``True`` ，表示返回周期窗口。如果为 ``False`` ，表示返回对称窗口。默认 ``True`` 。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        一维tensor。
