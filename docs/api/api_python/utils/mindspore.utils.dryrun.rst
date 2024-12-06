
.. py:function:: mindspore.utils.dryrun.set_simulation()

    用于设置dryrun功能开启。dryrun功能主要用于模拟大模型的实际运行，开启后可以在不占卡的情况下模拟出显存占用，编译信息等。
    在pynative场景下开启后，若存在从device取值到host场景，会打印出python调用栈日志，告知用户这些值是不准确的。

    返回：
        无。

    支持平台：
        ``Ascend``

    **样例**：

        >>> from mindspore.utils import dryrun
        >>> import os
        >>> dryrun.set_simulation()
        >>> print(os.environ.get("MS_SIMULATION_LEVEL"))
        1

.. py:function:: mindspore.utils.dryrun.mock(mock_val, *args)
    在网络中若一些if判断需要用到实际执行值，虚拟执行无法获取，可以使用该接口返回模拟值。实际执行时可以获取到正确结果，返回执行值。

    返回：
        开启dryrun功能，返回静态模拟值（mock_val），否则返回实际执行结果（args）。

    支持平台：
        ``Ascend``

    **样例**：

        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> dryrun.set_simulation()
        >>> a = ms.Tensor(np.random.rand(3, 3).astype(np.float32))
        >>> if dryrun.mock(True, a[0, 0] > 0.5):
        ...     print("return mock_val: True.")
        return mock_val: True

        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, a[0, 0] > 0.5):
        ...     print("return real execution: True.")
        return real execution: True.

        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, (a > 0.5).any):
        ...     print("return real execution: True.")
        return real execution: True.
