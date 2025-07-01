mindspore.numpy.diff
====================

.. py:function:: mindspore.numpy.diff(a, n=1, axis=- 1, prepend=None, append=None)

    计算沿给定 `axis` 的n阶离散差分。

    一阶差分由 :math:`out[i] = a[i+1] - a[i]` 给出，沿给定 `axis`，通过迭代使用 `diff` 计算更高阶的差分。

    .. note::
        由于MindSpore不支持zero-shaped的Tensor，如果遇到空Tensor将引发ValueError。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **n** (int, 可选) - 差分阶数。 如果为零，输出将按原样返回。 默认值：1。
        - **axis** (int, 可选) - 取差分的轴。 默认为最后一个轴。 默认值：-1。
        - **prepend/append** (Tensor, 可选) - 在执行差分之前在 `axis` 的前/后向 `a` 填充的值。 如果是标量值，会沿着 `axis` 方向填充，填充宽度为1，在所有其他方向上与输入数组shape相同的数组。否则，除了指定的 `axis` ，其余维度和shape必须与输入数组 `a` 匹配。 默认值： ``None`` 。

    返回：
        n阶差分。 输出的shape除了在指定的 `axis` 方向上的维数比 `a` 小 `n` 外，其他与 `a` 相同。 输出的类型与 `a` 中任意两个元素之间差分的类型相同，在大多数情况下，与 `a` 的类型相同。

    异常：
        - **TypeError** - 如果输入的类型不符合上述指定类型。
        - **ValueError** - 如果 `n` < 0。