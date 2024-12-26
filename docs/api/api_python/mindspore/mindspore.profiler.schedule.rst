mindspore.profiler.schedule
===========================

.. py:class:: mindspore.profiler.schedule(*, wait: int, active: int, warm_up: int = 0, repeat: int = 0, skip_first: int = 0)

    该类用于获取每一步的操作。
    
    调度如下：

    .. code-block::

        (NONE)        (NONE)          (NONE)       (WARM_UP)       (RECORD)      (RECORD)     (RECORD_AND_SAVE)    None
        START------->skip_first------->wait-------->warm_up-------->active........active.........active----------->stop
                                      |                                                             |
                                      |                           repeat_1                          |
                                      ---------------------------------------------------------------
    
    Profiler将跳过前 ``skip_first`` 步，然后等待 ``wait`` 步，接着在接下来的 ``warm_up`` 步中进行预热，然后在接下来的 ``active`` 步中进行活动记录，然后从 ``wait`` 步开始重复循环。可选的循环次数由 ``repeat`` 参数指定， ``repeat`` 值为0表示循环将继续直到分析完成。


    关键字参数：
        - **wait** (int) - 预热阶段等待的步数。
        - **active** (int) - 活动阶段执行的步数。
        - **warm_up** (int, 可选) - 预热阶段执行的步数。默认值： ``0`` 。
        - **repeat** (int, 可选) - 重复次数。默认值： ``0`` 。
        - **skip_first** (int, 可选) - 跳过第一个步数。默认值： ``0`` 。

    异常：
        - **ValueError** - 参数step必须大于等于0。

    .. py:method:: to_dict()

        将schedule类转换为一个字典。

        返回：
            字典，schedule类的参数与对应的值。
