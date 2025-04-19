﻿mindspore.parallel.Layout
============================================================================

.. py:class:: mindspore.parallel.Layout(device_matrix, alias_name, rank_list=None)

    描述集群设备的拓扑抽象，用于张量分片在集群上的放置。

    .. note::
        - 仅在半自动并行或自动并行模式下有效。
        - `device_matrix` 的累乘结果必须等于一个 `pipeline stage` 中的设备数。
        - 当 `Layout` 来构建切分策略时，每个别名只允许用于一次张量的切分。

    参数：
        - **device_matrix** (tuple) - 描述设备排列的形状，其元素类型为 int 。
        - **alias_name** (tuple) - `device_matrix` 的每个轴的别名，其元素类型为字符串。使用 `interleaved_parallel` 作为别名时，会在其对应的切分维度将该算子在单卡内拆分为多个副本。
        - **rank_list** (list，可选) - 数据根据 `rank_list` 排布在设备上。默认 ``None``。

    异常：
        - **TypeError** - `device_matrix` 不是tuple类型。
        - **TypeError** - `alias_name` 不是tuple类型。
        - **TypeError** - `rank_list` 不是list类型。
        - **ValueError** - `device_matrix` 长度不等于 `alias_name` 长度。
        - **TypeError** - `device_matrix` 的元素不是 int 类型。
        - **TypeError** - `alias_name` 的元素不是 str 类型。
        - **TypeError** - `rank_list` 的元素不是 int 类型。
        - **ValueError** - `alias_name` 的元素是一个空的 str 。
        - **ValueError** - `alias_name` 的元素为 None 。
        - **ValueError** - `alias_name` 包含重复的元素。


    .. py:method:: to_dict()

        将Layout转换为词典。
