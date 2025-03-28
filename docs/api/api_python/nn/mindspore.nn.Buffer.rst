mindspore.nn.Buffer
===================

.. py:class:: mindspore.nn.Buffer(data, persistent=True)

    一种不被视为模型参数的Tensor。例如，BatchNorm的 `running_mean` 不是参数，而是Cell状态的一部分。

    Buffer是 :class:`~mindspore.Tensor` 的子类，在 :class:`~.nn.Cell` 中具有特殊的属性：当它们被赋值为Cell的属性时，会自动添加到
    Cell的buffer列表中，并会出现在 :func:`mindspore.nn.Cell.buffers` 迭代器中。直接赋值tensor不会有这样的效果，
    但可以使用 :func:`mindspore.nn.Cell.register_buffer` 方法将tensor显式地注册成buffer。

    参数：
        - **data** (Tensor) - buffer的张量数据。
        - **persistent** (bool, 可选) - buffer是否作为Cell的 :attr:`state_dict` 的一部分。默认 ``True`` 。
