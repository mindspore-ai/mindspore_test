mindspore.experimental.es.ESEmbeddingSmallTableLookup
=========================================================

.. py:class:: mindspore.experimental.es.ESEmbeddingSmallTableLookup(name, rank_id, rank_size, small_table_to_variable)

    提供小表特征值keys的查询功能。

    .. warning::
        这是一个实验性的EmbeddingService接口。

    参数：
        - **name** (str) - 小表名字。
        - **rank_id** (int) - 查询小表key时的rank id。
        - **rank_size** (int) - 查询小表是的rank size。
        - **small_table_to_variable** (dict[str, parameter]) - 存储小表信息的dict：key为小表名称，value为小表对应的参数信息。

    输入：
        - **ids_list** (Tensor) - 小表的每个特征keys。
