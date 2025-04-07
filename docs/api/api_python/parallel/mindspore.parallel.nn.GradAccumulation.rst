mindspore.parallel.nn.GradAccumulation
============================================================================

.. py:class:: mindspore.parallel.nn.GradAccumulation(network, micro_size)

    使能GradAccumulation实现梯度累加。

    参数：
        - **network** (Cell) - 将进行梯度累加的网络。
        - **micro_size** (int) - MicroBatchSize。

    异常：
        - **TypeError** - `network` 不是cell类型输入。
        - **TypeError** - `micro_size` 不是整数类型。
        - **ValueError** - `micro_size` 值异常，为0或者负数。
