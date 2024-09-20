mindspore.numpy.histogram_bin_edges
====================================

.. py:function:: mindspore.numpy.histogram_bin_edges(a, bins=10, range=None, weights=None)

    计算 ``histogram`` 函数需要使用的 ``bins`` 的边界值。

    .. note::
        ``bins`` 不支持string类型。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入数据。直方图是在展平的数组上计算的。
        - **bins** (Union[int, tuple, list, Tensor], 可选) - 指定分组的数量或具体边界。如果 ``bins`` 是一个int型数据，则它给定了分组数（默认为 ``10`` 个），分组是在给定范围的等宽分组。如果 ``bins`` 是序列，则它将定义每一个分组的边界（包括最右边），此时分组宽度可以是不等的。
        - **range** ((float, float), 可选) - 指定边界的最小值和最大值。如果未提供，则默认范围为 ``(a.min()，a.max())`` 。超出范围的值将被忽略。范围的第一个元素必须小于或等于第二个。
        - **weights** (Union[int, float, bool, list, tuple, Tensor], 可选) - 与 ``a`` shape相同的权重数组。其中的每个权重值赋给相应的 ``a`` 中的值。目前暂未被任何分组估计算法使用，未来可能被使用。默认值： ``None`` 。

    返回：
        Tensor，传递到  ``histogram`` 函数中的边界值。

    异常：
        - **TypeError** - 如果 ``bins`` 是一个数组且不是一维的。