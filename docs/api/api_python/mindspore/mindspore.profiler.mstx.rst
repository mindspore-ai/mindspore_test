mindspore.profiler.mstx
========================

.. py:class:: mindspore.profiler.mstx

    该类可在NPU上进行性能分析的标记和跟踪。该类提供了mark、range_start和range_end三个静态方法，用于在性能分析中添加标记点和区间。

    .. py:method:: mark(message: str, stream: Stream = None)
        :staticmethod:

        在性能分析中添加标记点。

        参数：
            - **message** (str) - 标记点的描述信息。
            - **stream** (Stream, 可选) - 用于异步执行的NPU流，期望类型：mindspore.runtime.Stream。默认值： ``None`` ，表示仅在host侧添加标记点，不在device侧的stream上添加标记点。

    .. py:method:: range_start(message: str, stream: Stream = None)
        :staticmethod:

        开始一个性能分析区间。

        参数：
            - **message** (str) - 区间的描述信息。
            - **stream** (Stream, 可选) - 用于异步执行的NPU流，期望类型：mindspore.runtime.Stream。默认值： ``None`` ，表示仅在host侧开始区间打点，不在device侧的stream上开始区间打点。

        返回：
            int，区间ID，用于range_end方法。

    .. py:method:: range_end(range_id: int)
        :staticmethod:

        结束一个性能分析区间。

        参数：
            - **range_id** (int) - 从range_start返回的区间ID。
