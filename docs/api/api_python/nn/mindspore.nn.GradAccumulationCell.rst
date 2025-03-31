mindspore.nn.GradAccumulationCell
=================================

.. py:class:: mindspore.nn.GradAccumulationCell(network, micro_size)

    将MiniBatch切分成更细粒度的MicroBatch，用于半自动/全自动并行模式下的梯度累加训练中。

    .. note::
        接口即将废弃，请使用接口 :class:`mindspore.parallel.nn.GradAccumulation` 代替。

    参数：
        - **network** (Cell) - 要修饰的目标网络。
        - **micro_size** (int) - MicroBatch大小。
