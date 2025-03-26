mindspore.parallel.nn.Pipeline
============================================================================

.. py:function:: mindspore.parallel.nn.Pipeline(net, micro_size, stage_config=None)

    指定流水线并行（pp）的micro_batch个数以及网络中各cell放到哪个stage去执行。

    参数：
        - **net** (Cell) - 将进行pp并行的网络。
        - **micro_size** (int) - MicroBatchsize。
        - **stage_config** (dict，可选) - 流水线并行对于每个cell的stage配置。默认值： ``None``。

    异常：
        - **TypeError** - `net` 不是cell类型输入。
        - **TypeError** - `micro_size` 不是整数类型。
        - **ValueError** - `micro_size` 值异常，为0或者负数。
        - **KeyError** - `dict` cell名称匹配异常，遍历当前 `net` 下所有 `cell` 仍有剩余的配置项。
