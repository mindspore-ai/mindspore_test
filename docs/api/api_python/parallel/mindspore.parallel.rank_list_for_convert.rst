mindspore.parallel.rank_list_for_convert
========================================

.. py:function:: mindspore.parallel.rank_list_for_convert(rank_id, src_strategy_file=None, dst_strategy_file=None)

    在对分布式Checkpoint转换的过程中，获取目标rank的Checkpoint文件所需的源Checkpoint文件rank列表。

    参数：
        - **rank_id** (int) - 待转换的Checkpoint的rank号。
        - **src_strategy_file** (str) - 源切分策略proto文件名，由 `AutoParallel.save_param_strategy_file(strategy_ckpt_save_file)` 接口存储下来的文件。当其为 ``None`` 时，表示切分策略为不切分。默认值： ``None`` 。
        - **dst_strategy_file** (str) - 目标切分策略proto文件名，由 `AutoParallel.save_param_strategy_file(strategy_ckpt_save_file)` 接口存储下来的文件。当其为 ``None`` 时，表示切分策略为不切分。默认值： ``None`` 。

    返回：
        转换得到rank_id的分布式Checkpoint所需要的卡号列表。

    异常：
        - **ValueError** - `src_strategy_file` 或者 `dst_strategy_file` 不是正确的切分策略proto文件。
        - **TypeError** - `src_strategy_file` 或者 `dst_strategy_file` 不是字符串。
        - **TypeError** - `rank_id` 不是一个整数。
