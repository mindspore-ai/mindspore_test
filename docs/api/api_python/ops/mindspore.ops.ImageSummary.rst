mindspore.ops.ImageSummary
==========================

.. py:class:: mindspore.ops.ImageSummary

    将图像保存到Summary文件。该算子必须和SummaryRecord或SummaryCollector一起使用，
    Summary文件的保存路径由SummaryRecord或SummaryCollector指定。

    在Ascend平台上的Graph模式下，可以通过设置环境变量 `MS_DUMP_SLICE_SIZE` 和 `MS_DUMP_WAIT_TIME` 解决该算子在调用比较密集的场景下执行失败的问题。

    输入：
        - **name** (str) - 输入变量的名称，不能是空字符串。
        - **value** (Tensor) - 图像数据的值，Tensor的rank必须为4。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
