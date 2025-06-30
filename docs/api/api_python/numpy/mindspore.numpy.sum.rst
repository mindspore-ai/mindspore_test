mindspore.numpy.sum
===================

.. py:function:: mindspore.numpy.sum(a, axis=None, dtype=None, keepdims=False, initial=None)

    返回指定轴上数组元素的总和。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 要计算和的元素。
        - **axis** (Union[None, int, tuple(int)], 可选) - 执行求和操作的单个或多个轴。 默认值： ``None`` 。如果为 `None` ，则求输入数组中所有元素的和。如果 `axis` 为负数，则从最后一个轴到第一个轴计数。 如果 `axis` 是元素为整数的tuple，则对tuple中指定的所有轴进行求和，而非前述的单个轴或所有轴。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。
        - **keepdims** (bool, 可选) - 如果设置为 `True` ，减少的轴在结果中保留为大小为1的维度。 若使用此选项，结果会广播到和输入Tensor同一个维度数。 如果传入默认值，则 `keepdims` 参数不会传递到ndarray子类的std方法中，而任何非默认值将会传递。 如果子类中方法未实现 `keepdims` ，则会引发异常。 默认值： ``False`` 。
        - **initial** (scalar, 可选) - 求和的初始值，如果为 ``None``，不会添加额外的初始值，从第一个元素的值开始累加。 默认值： ``None`` 。

    返回：
        Tensor。 具有与 `a` 相同shape的数组，去除了指定的轴。 如果 `a` 是 0-d 数组，或者 `axis` 为 `None`，则返回标量。如果指定了输出数组，则返回对 `out` 的引用。

    异常：
        - **TypeError** - 如果input不是类数组或 `axis` 不是int或元素为int的tuple，或如果 `keepdims` 不是整数或 `initial` 不是标量。
        - **ValueError** - 如果任何轴超出范围或存在重复的轴。