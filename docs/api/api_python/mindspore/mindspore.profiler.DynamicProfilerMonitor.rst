mindspore.profiler.DynamicProfilerMonitor
=========================================

.. py:class:: mindspore.profiler.DynamicProfilerMonitor(cfg_path, output_path="./dyn_profile_data", poll_interval=2, **kwargs)

    该类用于动态采集MindSpore神经网络性能数据。

    参数：
        - **cfg_path** (str) - 动态profile的json配置文件文件夹路径。要求该路径是能够被所有节点访问到的共享目录。json配置文件相关参数如下。

          - start_step (int, 必选) - 设置Profiler开始采集的步数，为相对值，训练的第一步为1。默认值-1，表示在整个训练流程不会开始采集。
          - stop_step (int, 必选) - 设置Profiler开始停止的步数，为相对值，训练的第一步为1，需要满足stop_step大于等于start_step。默认值-1，表示在整个训练流程不会开始采集。
          - aicore_metrics (int, 可选) - 设置采集AI Core指标数据，取值范围与Profiler一一对应。默认值-1，表示不采集AI Core指标，0代表PipeUtilization；1代表ArithmeticUtilization；2代表Memory；3代表MemoryL0；4代表MemoryUB；5代表ResourceConflictRatio；6代表L2Cache。
          - profiler_level (int, 可选) - 设置采集性能数据级别，0代表ProfilerLevel.Level0，1代表ProfilerLevel.Level1，2代表ProfilerLevel.Level2。默认值0，表示ProfilerLevel.Level0的采集级别。
          - activities (int, 可选) - 设置采集性能数据的设备，0代表CPU+NPU，1代表CPU，2代表NPU。默认值0，表示采集CPU+NPU的性能数据。
          - analyse_mode (int, 可选) - 设置在线解析的模式，对应mindspore.Profiler.analyse接口的analyse_mode参数，0代表"sync"，1代表"async"。默认值-1，表示不使用在线解析。
          - parallel_strategy (bool, 可选) - 设置是否采集并行策略性能数据，true代表采集，false代表不采集。默认值false，表示不采集并行策略性能数据。
          - with_stack (bool, 可选) - 设置是否采集调用栈信息，true代表采集，false代表不采集。默认值false，表示不采集调用栈。
          - data_simplification (bool, 可选) - 设置开启数据精简，true代表开启，false代表不开启。默认值true，表示开启数据精简。

        - **output_path** (str, 可选) - 动态profile的输出文件路径。默认值：``"./dyn_profile_data"`` 。
        - **poll_interval** (int, 可选) - 监控进程的轮询周期，单位为秒。默认值：``2``。

    异常：
        - **RuntimeError** - 创建监控进程失败次数超过最大限制。

    .. py:method:: step()

        用于在Ascend设备上，区分step收集和解析性能数据。

        异常：
            - **RuntimeError** - 如果 `start_step` 参数设置大于 `stop_step` 参数设置 。