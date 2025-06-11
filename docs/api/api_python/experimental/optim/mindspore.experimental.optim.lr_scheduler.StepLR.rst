mindspore.experimental.optim.lr_scheduler.StepLR
=================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

    在训练过程中，当调用 `StepLR.step()` 时，若当前epoch数是 `step_size` 的整数倍，则学习率会乘以 `gamma` 进行衰减。学习率的调整与优化器的参数更新是协同进行的，优化器依据当前调整后的学习率来执行参数优化操作。`StepLR` 对于学习率的衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **step_size** (int) - 学习率衰减的周期。
        - **gamma** (float，可选) -  学习率衰减的乘法因子。默认值: ``0.1``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值： ``-1``。