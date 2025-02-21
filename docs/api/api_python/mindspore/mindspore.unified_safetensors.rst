mindspore.unified_safetensors
==============================

.. py:function:: mindspore.unified_safetensors(src_dir, src_strategy_file, dst_dir, merge_with_redundancy=True, file_suffix=None, max_process_num=64, choice_func=None)

    将多个safetensors文件合并为一系列统一的safetensors文件。

    参数：
        - **src_dir** (str) - 源权重保存目录。
        - **src_strategy_file** (str) - 源权重切分策略文件。
        - **dst_dir** (str) - 目标保存目录。
        - **merge_with_redundancy** (bool, 可选) - 合并源权重文件是否是去冗余保存的safetensors文件。默认值是：``True``，合并的源权重文件是完整的。
        - **file_suffix** (str, 可选) - 指定合并safetensors的文件名后缀。默认值是：``None`` ，合并源权重目录下所有的safetensors文件。
        - **max_process_num** (int) - 最大进程数。默认值：64。
        - **choice_func** (callable) - 可调用的函数，用于筛选参数或者修改参数名，函数的返回值必须为str或者bool类型。默认值：``None``。

    异常：
        - **ValueError** - 如果某个rank的safetensors文件丢失。
