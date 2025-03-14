mindspore.profiler.profile
===========================

.. py:class:: mindspore.profiler.profile(ctivities: list = None, with_stack: bool = False, profile_memory: bool = False, data_process: bool = False, parallel_strategy: bool = False, start_profile: bool = True, hbm_ddr: bool = False, pcie: bool = False, sync_enable: bool = True, schedule: Schedule = None, on_trace_ready: Optional[Callable[..., Any]] = None, experimental_config: Optional[_ExperimentalConfig] = None)

    MindSpore用户能够通过该类对神经网络的性能进行采集。可以通过导入该类初始化profile对象，
    使用 `profile.start()` 开始分析，并使用 `profile.stop()` 停止收集并分析结果。可通过
    `MindStudio Insight <https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann>`_
    工具可视化分析结果。目前，profile支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

    参数：
        - **start_profile** (bool, 可选) - 该参数控制是否在Profiler初始化的时候开启数据采集。默认值： ``True`` 。
        - **activities** (list, 可选) - 表示需要收集的性能数据类型。默认值： ``[ProfilerActivity.CPU, ProfilerActivity.NPU]`` 。

          - ProfilerActivity.CPU：收集MindSpore框架数据。
          - ProfilerActivity.NPU：收集CANN软件栈和NPU数据。
          - ProfilerActivity.GPU：收集GPU数据。
        - **schedule** (schedule, 可选) - 设置采集的动作策略，由schedule类定义，需要配合step接口使用，默认值： ``None`` 。
        - **on_trace_ready** (Callable, 可选) - 设置当性能数据采集完成时，执行的回调函数。默认值： ``None`` 。
        - **profile_memory** (bool, 可选) -（仅限Ascend）表示是否收集Tensor内存数据。当值为 ``True`` 时，收集这些数据。使用该参数时， `activities` 必须设置为 ``[ProfilerActivity.CPU, ProfilerActivity.NPU]`` 。在图编译等级为O2时收集算子内存数据，需要从第一个step开始采集。默认值： ``False`` ，该参数目前采集的算子名称不完整。将在后续版本修复，建议使用环境变量 ``MS_ALLOC_CONF`` 代替。
        - **with_stack** (bool, 可选) - （仅限Ascend）表示是否收集Python侧的调用栈的数据，此数据在timeline中采用火焰图的形式呈现，使用该参数时， `activities` 必须包含 ``ProfilerActivity.CPU`` 。默认值： ``False`` 。
        - **hbm_ddr** (bool, 可选) -（仅限Ascend）是否收集片上内存/DDR内存读写速率数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **pcie** (bool, 可选) -（仅限Ascend）是否收集PCIe带宽数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **data_process** (bool, 可选) -（Ascend/GPU）表示是否收集数据准备性能数据，默认值： ``False`` 。
        - **parallel_strategy** (bool, 可选) -（仅限Ascend）表示是否收集并行策略性能数据，默认值： ``False`` 。
        - **sync_enable** (bool, 可选) -（仅限GPU）Profiler是否用同步的方式收集算子耗时，默认值： ``True`` 。

          - True：同步方式，在把算子发送到GPU之前，在CPU端记录开始时间戳。然后在算子执行完毕返回到CPU端后，再记录结束时间戳。算子耗时为两个时间戳的差值。
          - False：异步方式，算子耗时为从CPU发送到GPU的耗时。这种方式能减少因增加Profiler对整体训练时间的影响。
        - **experimental_config** (_ExperimentalConfig, 可选) - 可扩展的参数可以在此配置。

    异常：
        - **RuntimeError** - 当CANN的版本与MindSpore版本不匹配时，MindSpore无法解析生成的ascend_job_id目录结构。

    .. py:method:: add_metadata(key: str, value: str)

        上报自定义metadata键值对数据。

        参数：
            - **key** (str) - metadata键值对的key。
            - **value** (str) - metadata键值对的value。

    .. py:method:: add_metadata_json(key: str, value: str)

        上报自定义metadata键值对value为json字符串数据。

        参数：
            - **key** (str) - metadata键值对的key。
            - **value** (str) - metadata键值对的value，格式为json字符串。
    .. py:method:: start()

        开启profile数据采集。可以按条件开启profile。

        异常：
            - **RuntimeError** - profile已经开启。
            - **RuntimeError** - 如果 `start_profile` 参数未设置或设置为 ``True`` 。

    .. py:method:: step()

        用于在Ascend设备上，通过schedule和on_trace_ready区分步骤收集和解析性能数据。

        异常：
            - **RuntimeError** - 如果 `start_profile` 参数未设置或profile未开启。
            - **RuntimeError** - 如果 `schedule` 参数未设置。

    .. py:method:: stop()

        停止profile。可以按条件停止profile。

        异常：
            - **RuntimeError** - profile没有开启。

.. py:function:: analyse(profiler_path: str, max_process_number: int = os.cpu_count() // 2, pretty=False, step_list=None, data_simplification=True)

        离线分析训练的性能数据，性能数据采集结束后调用。

        参数：
            - **profiler_path** (str) - 需要进行离线分析的profiling数据路径，指定到profiler上层目录。
            - **max_process_number** (int, 可选) - 最大进程数，默认值为 ``os.cpu_count() // 2`` 。
            - **pretty** (bool, 可选) - 对json文件进行格式化处理。该参数默认值为 ``False``，即不进行格式化。
            - **step_list** (list, 可选) - 只解析指定step的性能数据，指定的step必须是连续的整数。该参数默认值为 ``None``，即进行全解析。
            - **data_simplification** (bool, 可选) - 数据精简开关功能。默认值为 ``True``，即开启数据精简。