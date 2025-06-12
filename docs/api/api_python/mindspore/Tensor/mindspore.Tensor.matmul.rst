mindspore.Tensor.matmul
=======================

.. py:method:: mindspore.Tensor.matmul(tensor2) -> Union[Tensor, numbers.Number]

    计算两个数组的矩阵乘积。

    .. note::
        - 不支持NumPy参数 `out` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        - `self` 和 `tensor2` 的数据类型必须一致。
        - 在Ascend平台上， `self` 和 `tensor2` 的维度必须在1到6之间。
        - 在GPU平台上， `self` 和 `tensor2` 支持的数据类型是ms.float16和ms.float32。

    参数：
        - **tensor2** (Tensor) - 输入Tensor，不支持Scalar， `self` 的最后一维度和 `tensor2` 的倒数第二维度相等，且 `self` 和 `tensor2` 彼此支持广播。

    返回：
        Tensor或Scalar，输入的矩阵乘积。仅当 `self` 和 `tensor2` 为一维向量时，输出为Scalar。

    异常：
        - **TypeError** - `self` 的dtype和 `tensor2` 的dtype不一致。
        - **ValueError** -  `self` 的最后一维度和 `tensor2` 的倒数第二维度不相等，或者输入的是Scalar。
        - **ValueError** - `self` 和 `tensor2` 彼此不能广播。
        - **RuntimeError** - 在Ascend平台上， `self` 或 `tensor2` 的维度小于1或大于6。