mindspore.profiler._ExperimentalConfig
=======================================

.. py:class:: mindspore.profiler._ExperimentalConfig(profiler_level: ProfilerLevel = ProfilerLevel.Level0, aic_metrics: AicoreMetrics = AicoreMetrics.AiCoreNone, l2_cache: bool = False, mstx: bool = False, data_simplification: bool = True, export_type: list = None, mstx_domain_include: list = None, mstx_domain_exclude: list = None, sys_io: bool = False, sys_interconnection: bool = False, host_sys: list = None)

    在使用profile进行模型性能数据采集时，配置可扩展的参数。

    参数：
        - **profiler_level** (ProfilerLevel, 可选) - （仅限Ascend）表示采集性能数据级别。默认值：``ProfilerLevel.Level0`` 。

          - ProfilerLevel.LevelNone：该设置仅在开启mstx时生效，表示不采集device侧的任何算子数据。
          - ProfilerLevel.Level0：最精简的采集性能数据级别，采集计算类算子的耗时数据和通信类大算子的基础数据。
          - ProfilerLevel.Level1：在Level0的基础上额外采集CANN层中AscendCL数据、AICORE性能数据以及通信类小算子数据。
          - ProfilerLevel.Level2：在Level1的基础上额外采集CANN层中GE和Runtime数据。
        - **aic_metrics** (AicoreMetrics, 可选) - （仅限Ascend）收集的AICORE性能数据类型，使用此参数时， `activities` 必须包含 ``ProfilerActivity.NPU`` ，且值必须包含在AicoreMetrics枚举值中，当profiler_level为Level0，默认值为： ``AicoreMetrics.AiCoreNone`` ；profiler_level为Level1或Level2，默认值为：``AicoreMetrics.PipeUtilization``，当每种类型包含的数据项如下：

          - AicoreMetrics.AiCoreNone：不收集任何AICORE数据。
          - AicoreMetrics.ArithmeticUtilization：包含mac_fp16/int8_ratio、vec_fp32/fp16/int32_ratio、vec_misc_ratio等。
          - AicoreMetrics.PipeUtilization：包含vec_ratio、mac_ratio、scalar_ratio、mte1/mte2/mte3_ratio、icache_miss_rate等。
          - AicoreMetrics.Memory：包含ub\_read/write_bw、l1_read/write_bw、l2_read/write_bw、main_mem_read/write_bw等。
          - AicoreMetrics.MemoryL0：包含l0a_read/write_bw、l0b_read/write_bw、l0c_read/write_bw等。
          - AicoreMetrics.ResourceConflictRatio：包含vec_bankgroup/bank/resc_cflt_ratio等。
          - AicoreMetrics.MemoryUB：包含ub\_read/write_bw_mte、 ub\_read/write_bw_vector、 ub\_read/write_bw_scalar等。
          - AicoreMetrics.L2Cache：包含write_cache_hit、 write_cache_miss_allocate、 r0_read_cache_hit、 r1_read_cache_hit等。本功能仅支持Atlas A2 训练系列产品。
          - AicoreMetrics.MemoryAccess：主存以及L2 Cache的存访带宽和存量统计。
        - **l2_cache** (bool, 可选) - （仅限Ascend）是否收集L2 Cache数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。该采集项在ASCEND_PROFILER_OUTPUT文件夹下生成l2_cache.csv文件。在O2模式下，仅支持schedule配置中wait和skip_first参数都为0的场景。
        - **mstx** (bool, 可选) - （仅限Ascend）是否收集MSTX数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **data_simplification** (bool, 可选) - （仅限Ascend）是否开启数据精简，开启后仅保留profiler的交付件以及PROF_XXX目录下的原始性能数据，以节省空间。默认值: ``True`` 。
        - **export_type** (list, 可选) - （仅限Ascend）要导出的数据类型，支持同时导出db和text格式，默认值： ``None``，表示导出text类型数据。

          - ExportType.Text：导出text类型的数据。
          - ExportType.Db：导出db类型的数据。
        - **mstx_domain_include** (list, 可选) - （仅限Ascend）mstx开关打开时设置使能的domain名称集合，且名称必须是str类型。默认值：``[]`` ，表示不使用该参数控制domain。该参数与mstx_domain_exclude参数互斥，不能同时设置。如果都设置，只有mstx_domain_include参数生效。
        - **mstx_domain_exclude** (list, 可选) - （仅限Ascend）mstx开关打开时设置不使能的domain名称集合，且名称必须是str类型。默认值：``[]`` ，表示不使用该参数控制domain。
        - **sys_io** (bool, 可选) - （仅限Ascend）是否收集NIC和RoCE数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **sys_interconnection** (bool, 可选) - （仅限Ascend）是否收集系统互连数据，包括集合通信带宽数据（HCCS）、PCIe数据以及片间传输带宽信息，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **host_sys** (list, 可选) - 表示采集host侧系统类调用类、存储类、cpu占用率数据。默认值： ``[]`` ，表示不采集host侧系统类数据。

          - HostSystem.CPU：收集进程级别的CPU利用率。
          - HostSystem.MEM：收集进程级别的内存利用率。
          - HostSystem.DISK：收集进程级别的磁盘I/O利用率。需要将 :class:`mindspore.profiler.profile` 中的 `start_profile` 参数设置为 ``False``。
          - HostSystem.NETWORK：收集系统级别的网络I/O利用率。
          - HostSystem.OSRT：收集系统级别系统调用栈数据，注意在配置用户权限时，需要替换msprof_data_collection.sh内容为 `msprof_data_collection.sh <https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python/mindspore/script/msprof_data_collection.sh>`_。需要将 :class:`mindspore.profiler.profile` 中的 `start_profile` 参数设置为 ``False``。

    异常：
        - **RuntimeError** - 当CANN的版本与MindSpore版本不匹配时，MindSpore无法解析生成的ascend_job_id目录结构。