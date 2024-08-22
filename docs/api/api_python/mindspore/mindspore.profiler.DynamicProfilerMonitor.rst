mindspore.profiler.DynamicProfilerMonitor
=========================================

.. py:class:: mindspore.profiler.DynamicProfilerMonitor(cfg_path, output_path="./dyn_profile_data", poll_interval=2)
    该类用于动态采集MindSpore神经网络性能数据。

    参数：
        - **cfg_path** (str) - 动态profile的json配置文件文件夹路径。要求该路径是能够被所有节点访问到的共享目录。
        - **output_path** (str, 可选) - 动态profile的输出文件路径。默认值：``"./dyn_profile_data"`` 。
        - **poll_interval** (int, 可选) - 监控进程的轮询周期，单位为秒。默认值：``2``。

    异常：
        - **RuntimeError** - 创建监控进程失败次数超过最大限制。