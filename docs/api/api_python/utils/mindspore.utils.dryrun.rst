
.. py:function:: mindspore.utils.dryrun.set_simulation()

    用于设置dryrun功能开启。dryrun功能主要用于模拟大模型的实际运行，开启后可以在不占卡的情况下模拟出显存占用，编译信息等。
    在PyNative场景下开启后，若存在从device取值到host的场景，会打印出Python调用栈日志，告知用户这些值是不准确的。

.. py:function:: mindspore.utils.dryrun.mock(mock_val, *args)

    在网络中若一些if判断需要用到实际执行值，虚拟执行无法获取，可以使用该接口返回模拟值。实际执行时可以获取到正确结果，返回执行值。

    参数：
        - **mock_val** (Union[Value, Tensor]) - 虚拟执行场景下，想要输出的值。
        - **args** (Union[Value, function]) - 期望被mock掉的值，可以是函数或者变量。

    返回：
        开启dryrun功能，返回静态模拟值（mock_val），否则返回实际执行结果（args）。
