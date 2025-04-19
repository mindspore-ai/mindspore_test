mindspore.parallel.nn.MicroBatchInterleaved
================================================

.. py:class:: mindspore.parallel.nn.MicroBatchInterleaved(network, interleave_num=2)

    实现静态图并行多副本拆分功能，使得计算及通信能并发。
    使用场景：当在半自动模式以及网络中存在模型并行时，第1份切片数据前向计算的同时，第2份数据将会进行模型并行的通信，以此来达到通信计算并发的性能加速。

    参数：
        - **network** (Cell) - 需要封装的网络。
        - **interleave_num** (int，可选) - batch size的拆分份数，默认值： ``2`` 。

    输入：
        tuple[Tensor]，与传入的 `network` 的输入一致。

    输出：
        被封装后的网络。传入的 `network` 的输出只能是单个Tensor。
