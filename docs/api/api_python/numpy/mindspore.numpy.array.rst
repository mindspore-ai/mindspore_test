mindspore.numpy.array
=================================

.. py:function:: mindspore.numpy.array(obj, dtype=None, copy=True, ndmin=0)

    该函数接受一个类似数组的对象创建Tensor。

    参数：
        - **obj** (Union[int, float, bool, list, tuple]) - 可以转换为Tensor的任何形式的输入数据。 包含 ``int, float, bool, Tensor, list, tuple`` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor数据类型可以是 ``np.int32`` 或 ``int32`` 。如果 ``dtype`` 为 ``None`` ，则将从 ``obj`` 推断出新Tensor的数据类型。默认值： ``None`` 。
        - **copy** (bool) - 如果为 ``True`` ，输入的对象会被拷贝；如果为 ``False`` ，只有在需要时对象才会被拷贝，默认值： ``True`` 。
        - **ndmin** (int) - 用于指定输出的Tensor应具有的最小维度数，将根据需要对其shape进行预处理。默认值： ``0`` 。

    返回：
        Tensor，具有指定数据类型。

    异常：
        - **TypeError** - 如果没有按照上述内容给定的数据类型输入参数。
        - **ValueError** - 如果输入 ``obj`` 在不同的维度上具有不同的大小。