mindspore.dataset.MappableDataset.add_sampler
==============================================

.. py:method:: mindspore.dataset.MappableDataset.add_sampler(new_sampler)

    为当前数据集添加子采样器。

    .. note::
        被添加的sampler如果有 `shuffle` 属性，其值必须是 ``Shuffle.GLOBAL`` ，且原sampler的 `shuffle` 属性值不能是 ``Shuffle.PARTIAL`` 。

    参数：
        - **new_sampler** (Sampler) - 待添加的子采样器。
