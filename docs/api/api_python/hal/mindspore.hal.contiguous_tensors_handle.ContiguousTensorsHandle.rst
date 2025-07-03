mindspore.hal.contiguous_tensors_handle.ContiguousTensorsHandle
===============================================================

.. py:class:: mindspore.hal.contiguous_tensors_handle.ContiguousTensorsHandle(tensor_list, enable_mem_align=True)

    连续内存管理器。

    参数：
        - **tensor_list** (list[Tensor], tuple[Tensor]) - 需要申请连续内存的Tensor列表。
        - **enable_mem_align** (bool，可选) - 是否启用内存对齐功能。暂不支持 ``False``。默认 ``True``。

    返回：
        ContiguousTensorsHandle，一个连续内存管理器。

    .. py:method:: slice_by_tensor_index(start=None, end=None)

        返回根据tensor列表的index切片出的连续内存。

        参数：
            - **start** (int, None) - 起始位置。默认 ``None``。
            - **end** (int, None) - 结束位置。默认 ``None``。

        返回：
            Tensor
