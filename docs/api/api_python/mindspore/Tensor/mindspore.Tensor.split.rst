mindspore.Tensor.split
=======================

.. py:method:: mindspore.Tensor.split(split_size_or_sections, axis=0)

    根据指定的轴将输入Tensor切分成块。

    参数：
        - **split_size_or_sections** (Union[int, tuple(int), list(int)]) - 如果 `split_size_or_sections` 是int类型，输入Tensor将被均匀的切分成块，每块的大小为 `split_size_or_sections` ，若 `tensor.shape[axis]` 不能被 `split_size_or_sections` 整除，最后一块大小将小于 `split_size_or_sections` 。如果 `split_size_or_sections` 是个list类型，输入Tensor将沿 `axis` 轴被切分成len(split_size_or_sections)块，大小为 `split_size_or_sections` 。
        - **axis** (int，可选) - 指定分割轴。默认值： ``0`` 。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - 如果 `axis` 不是int类型。
        - **ValueError** - 如果 `axis` 超出取值范围 :math:`[-tensor.ndim, tensor.ndim)` 。
        - **TypeError** - `split_size_or_sections` 中的每个元素不是int、tuple(int)或者list(int)。
        - **TypeError** - `split_size_or_sections` 的数据类型不是int、tuple(int)或者list(int)。
        - **ValueError** - `split_size_or_sections` 的和不等于 `x.shape[axis]` 。
