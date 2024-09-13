mindspore.numpy.column_stack
=================================

.. py:function:: mindspore.numpy.column_stack(tup)

    将一维Tensor作为列堆叠为二维Tensor。二维Tensor保持原样堆叠，如同 ``np.hstack`` 。

    参数：
        - **tup** (Union[Tensor, tuple, list]) - 一组一维或二维Tensor。它们必须具有相同的shape，除了要连接的轴外。

    返回：
        二维Tensor，由给定的Tensor堆叠而成。

    异常：
        - **TypeError** - 如果 ``tup`` 不是Tensor、list或tuple。
        - **ValueError** - 如果 ``tup`` 为空。