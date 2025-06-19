mindspore.hal.contiguous_tensors_handle.combine_tensor_list_contiguous
======================================================================

.. py:function:: mindspore.hal.contiguous_tensors_handle.combine_tensor_list_contiguous(tensor_list, enable_mem_align=True)

    返回一个连续内存管理器，在该内存管理器中，申请了连续内存，并提供切片功能。

    参数：
        - **tensor_list** (list[Tensor], tuple[Tensor]) - 需要申请连续内存的Tensor列表。
        - **enable_mem_align** (bool，可选) - 是否启用内存对齐功能。暂不支持False。默认 ``True``。

    返回：
        ContiguousTensorsHandle，一个连续内存管理器。
