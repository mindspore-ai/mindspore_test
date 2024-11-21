mindspore.common.parameter.no_init_parameters
=============================================

.. py:function:: mindspore.common.parameter.no_init_parameters

    加载checkpoint的场景下网络实例化中parameter会实例化并占用物理内存，加载checkpoint会替换parameter值，
    使用该接口在网络实例化时用装饰器给当前Cell里所有parameter添加一个属性： `init_param` ，并设为 `init_param=False` ，
    检测到 `init_param=False` 时跳过parameter初始化，加载checkpoint时从checkpoint给parameter赋值，可优化性能和减少物理内存。

    .. note::
        只能跳过使用 `initializer` 创建的parameter的初始化，由 `Tensor` 或 `numpy` 创建的parameter无法跳过。
