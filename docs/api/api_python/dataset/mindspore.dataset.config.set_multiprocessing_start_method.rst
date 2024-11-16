mindspore.dataset.config.set_multiprocessing_start_method
=========================================================

.. py:function:: mindspore.dataset.config.set_multiprocessing_start_method(start_method='fork')

    设置启动数据预处理子进程方式的全局配置。

    此设置将影响：GeneratorDataset自定义数据集、map操作和batch操作子进程的启动方式。

    参数：
        - **start_method** (str, 可选) - 数据预处理阶段子进程的启动方式。默认值： ``'fork'`` 。
          可选值：['fork', 'spawn']。

    异常：
        - **TypeError** - `start_method` 不是str类型。
        - **ValueError** - `start_method` 不是 ``'fork'`` 或者 ``'spawn'`` 。
