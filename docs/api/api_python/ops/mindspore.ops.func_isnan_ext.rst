mindspore.ops.isnan_ext
==========================

.. py:function:: mindspore.ops.isnan_ext(input)

    返回一个新的张量，其布尔元素表示输入的每个元素是否为 :math:`Nan` 。当复数的实部或虚部为 :math:`Nan` 时，该复数也被视为 :math:`Nan`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，返回一个布尔类型的Tensor，其中，当输入为 :math:`Nan` 时，Tensor值为 ``true``，否则为 ``false``。
