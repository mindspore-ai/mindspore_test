mindspore.ops.combinations
==========================

.. py:function:: mindspore.ops.combinations(input, r=2, with_replacement=False)

    返回输入tensor中所有长度为 `r` 的子序列。

    当 `with_replacement` 为 ``False`` 时，功能与Python的 `itertools.combinations` 类似；当为 ``True`` 时，功能与 `itertools.combinations_with_replacement` 一致。

    参数：
        - **input** (Tensor) - 一维输入tensor。
        - **r** (int，可选) - 进行组合的子序列长度。默认 ``2`` 。
        - **with_replacement** (bool，可选) - 是否允许组合存在重复值。默认 ``False`` 。

    返回：
        Tensor