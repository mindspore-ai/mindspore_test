mindspore.mint.empty_like
=========================

.. py:function:: mindspore.mint.empty_like(input, *, dtype=None, device=None)

    创建一个未初始化的Tesnor，shape和 `input` 相同，dtype由 `dtype` 决定，Tensor使用的内存由 `device` 决定。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype` 。如果为 ``None`` ，那么将会使\
          用 `input` 的dtype。默认值： ``None`` 。
        - **device** (string, 可选) - 指定Tensor使用的内存来源。当前支持 ``CPU`` 和 ``Ascend`` 。如果为 ``None`` ，
          那么将会使用 ``input`` 的内存来源，如果 ``input`` 没有申请内存，将会使用 `mindspore.set_device` 设置的值。
          默认值 ``None`` 。

    返回：
        Tensor，具有与 `input` 相同的shape，但是数据内容没有初始化（可能是任意值）。

    异常：
        - **TypeError** - `input` 不是Tensor。
