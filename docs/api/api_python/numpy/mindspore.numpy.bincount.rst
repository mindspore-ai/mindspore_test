mindspore.numpy.bincount
========================

.. py:function:: mindspore.numpy.bincount(x, weights=None, minlength=0, length=None)

    计算数组中非负int值的出现次数。结果返回一个桶(大小为1)数组，桶数组长度比 `x` 中的最大值大1。如果指定了 `minlength` ，输出数组至少会有 `minlength` 个桶(如果指定了该参数，根据 `x` 的长度，可能会让输出数组更长)。 每个桶记录了其索引值在 `x` 中出现的次数。如果指定了 `weights` ，则输入数组会根据权重加权，即如果在位置 `i` 处找到了值 `n` ，则以 ``out[n] += weight`` 代替 ``out[n] += 1`` 。

    .. note::
        附加的参数 `length` 定义了桶的数量(覆盖 ``x.max() + 1`` )，在 `graph` 模式下必须提供该参数。 如果 `x` 包含了负值，不会引发错误，负值将被视为0。

    参数：
        - **x** (Union[list, tuple, Tensor]) - 1-d输入数组。
        - **weights** (Union[int, float, bool, list, tuple, Tensor]，可选) - 权重，与 `x` 的shape相同的数组。 默认值：0。
        - **minlength** (int, 可选) - 输出数组的最小桶数。 默认值： `0` 。
        - **length** (int, 可选) - 桶的数量。 默认值： ``None`` 。

    返回：
        Tensor，即输入数组分桶后的结果。 输出的长度等于 ``np.amax(x) + 1`` 。

    异常：
        - **ValueError** - 如果 `x` 不是一维的，或 `x` 和 `weights` 的shape不同。