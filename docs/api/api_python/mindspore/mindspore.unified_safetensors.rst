mindspore.unified_safetensors
==============================

.. py:function:: mindspore.unified_safetensors(src_dir, src_strategy_file, dst_dir)

    将多个safetensors文件合并为一系列统一的safetensors文件。

    参数：
        - **src_dir** (str) - 源权重保存目录。
        - **src_strategy_file** (str) - 源权重切分策略文件。
        - **dst_dir** (str) - 目标保存目录。

    异常：
        - **ValueError** - 如果某个rank的safetensors文件丢失。
