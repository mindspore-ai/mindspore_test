mindspore.ops.ScalarSummary
============================

.. py:class:: mindspore.ops.ScalarSummary

    将标量数据保存到Summary文件。必须和 :class:`mindspore.SummaryRecord` 或 :class:`mindspore.SummaryCollector` 一起使用，
    Summary文件的保存路径由SummaryRecord或SummaryCollector指定。

    在Ascend平台上的Graph模式下，可以通过设置环境变量 `MS_DUMP_SLICE_SIZE` 和 `MS_DUMP_WAIT_TIME` 解决在密集调用场景下，该算子执行失败的问题。

    输入：
        - **name** (str) - 输入标量的名称，不能是空字符串。
        - **value** (Tensor) - 标量数据的值，维度必须为0或者1。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
        - **ValueError** - 如果 `value` 的维度大于1。
