mindspore.device_context.gpu.op_precision.conv_allow_tf32
=========================================================

.. py:function:: mindspore.device_context.gpu.op_precision.conv_allow_tf32(value)

    是否开启卷积在cuDNN下的TF32张量核计算。
    详细信息请查看 `nvidia关于CUBLAS_COMPUTE_32F_FAST_TF32的说明 <https://docs.nvidia.com/cuda/cublas/index.html>`_。

    参数：
        - **value** (bool) - 是否开启卷积在cuDNN下的TF32张量核计算。如未配置，框架默认为 ``True`` 。

    支持平台：
        ``GPU``