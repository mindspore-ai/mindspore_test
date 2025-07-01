mindspore.numpy.searchsorted
============================

.. py:function:: mindspore.numpy.searchsorted(a, v, side='left', sorter=None)

    找到每个元素的插入索引，使得插入后的数组保持原有升降序。在排序过的数组 `a` 中找到索引，使得如果在这些索引之前插入 `v` 中的相应元素， `a` 的顺序能够保持。

    参数：
        - **a** (Union[list, tuple, Tensor]) - 1-D输入数组。 如果 `sorter` 为 None，则 `a` 必须按升序排序，否则 `sorter` 必须是一个排序索引数组。
        - **v** (Union[int, float, bool, list, tuple, Tensor]) - 要插入 `a` 的值。
        - **side** ('left', 'right', 可选) - 如果为  `'left'` （默认值），则返回第一个合适位置的索引。 如果为 `'right'` ，则返回最后一个合适位置的索引。如果没有合适的索引，则返回 0 或 N (其中 N 为 `a` 的长度)。
        - **sorter** (Union[int, float, bool, list, tuple, Tensor]) - 可选择输入一个一维整数索引数组，用于将数组 `a` 排序为升序。通常是 `argsort` 的结果。默认值: `None` 。

    返回：
        Tensor，与 `v` shape相同的元素为插入点的数组。

    异常：
        - **ValueError** - 如果 `side` 或 `sorter` 参数无效。