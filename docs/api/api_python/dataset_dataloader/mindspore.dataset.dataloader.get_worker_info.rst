mindspore.dataset.dataloader.get_worker_info
============================================

.. py:function:: mindspore.dataset.dataloader.get_worker_info()

    获取当前DataLoader工作进程的信息。

    信息包括：

    - **worker_id** (int)：当前工作进程的ID。
    - **num_workers** (int)：工作进程的总数。
    - **seed** (int)：当前工作进程使用的随机种子。此值由主进程生成的基础种子和工作进程的ID确定。
    - **dataset** (Dataset)：从主进程复制到当前工作进程的数据集对象。

    如果当前进程不是DataLoader工作进程，则返回 ``None``。

    返回：
        Union[WorkerInfo, None]，当前DataLoader工作进程的信息。
