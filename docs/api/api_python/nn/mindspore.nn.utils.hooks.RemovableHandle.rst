mindspore.nn.utils.hooks.RemovableHandle
========================================

.. py:class:: mindspore.nn.utils.hooks.RemovableHandle(hooks_dict: Any, *, extra_dict: Any = None)

    提供一个移除钩子功能的句柄。

    参数：
        - **hooks_dict** (dict) - 一个以钩子 `id` 为索引的钩子字典。

    关键字参数：
        - **extra_dict** (Union[dict, list[dict]]，可选) - 一个额外的字典或字典列表，当 `hooks_dict` 中的相同键被移除时，这些键也会从该字典或字典列表中删除。

