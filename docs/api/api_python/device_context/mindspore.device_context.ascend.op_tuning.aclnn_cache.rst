mindspore.device_context.ascend.op_tuning.aclnn_cache
======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aclnn_cache(enable_global_cache=False, cache_queue_length=10000)

    配置aclnn单算子缓存参数。

    参数：
        - **enable_global_cache** (bool) - 配置全局缓存，默认为False，此参数只在GRAPH模式下生效。

          - ``"False"``: 关闭全局缓存。
          - ``"True"``: 开启全局缓存。
        - **cache_queue_length** (int，可选) - 配置aclnn缓存队列的长度，默认值：``10000``。
