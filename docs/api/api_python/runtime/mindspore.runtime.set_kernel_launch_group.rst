mindspore.runtime.set_kernel_launch_group
===========================================

.. py:function:: mindspore.runtime.set_kernel_launch_group(thread_num=2, kernel_group_num=8)

    kbk支持算子批量并行下发接口，支持开启并行下发，并配置并行数。

    参数：
        - **thread_num** (int) - 并发线程数，默认值2，一般不建议增加。和现有接口mindspore.runtime.dispatch_threads_num配置的线程数相互独立。
        - **kernel_group_num** (int) - 算子分组总数量，默认值8，每线程 kernel_group_num/thread_num个group.
