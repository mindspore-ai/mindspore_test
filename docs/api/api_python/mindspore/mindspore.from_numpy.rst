mindspore.from_numpy
=======================

.. py:function:: mindspore.from_numpy(array)

    将Numpy数组转换为张量。

    当数据为非C连续时，数据会被拷贝成C连续数据后再创建张量，否则会通过不复制数据的方式将Numpy数组转换为张量。

    参数：
        - **array** (numpy.array) - 输入数组。

    返回：
        与输入的张量具有相同的数据类型的Tensor。