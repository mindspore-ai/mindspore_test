mindspore.set_deterministic
============================

.. py:function:: mindspore.set_deterministic(deterministic)

    是否开启确定性计算。

    当开启确定性计算功能时，算子在相同的硬件和输入下，多次执行将产生相同的输出。但启用确定性计算往往导致算子执行变慢。

    在分布式场景下，建议用户在调用接口 :func:`mindspore.communication.init` 前设置确定性计算，以保证使能全局通信域上的通信算子确定性。

    框架默认不开启确定性计算。
    
    参数：
        - **deterministic** (bool) - 是否开启确定性计算。
