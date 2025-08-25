mindspore.dataset.dataloader.BatchSampler
=========================================

.. py:class:: mindspore.dataset.dataloader.BatchSampler(sampler, batch_size, drop_last)

    每次生成一个mini-batch索引的采样器。

    参数：
        - **sampler** (Union[Sampler, Iterable]) - 用于生成索引的采样器。
        - **batch_size** (int) - mini-batch的大小。
        - **drop_last** (bool) - 如果最后一批数据小于 `batch_size` ，是否舍弃该批次。
