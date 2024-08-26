mindspore.experimental.es.EsEmbeddingLookup
===================================================

.. py:class:: mindspore.experimental.es.EsEmbeddingLookup(table_id, es_initializer, embedding_dim, max_key_num, optimizer_mode=None, optimizer_params=None, es_filter=None, es_padding_key=None, es_completion_key=None)

    提供大表特征值keys的查询功能。

    .. warning::
        这是一个实验性的EmbeddingService接口。

    参数：
        - **table_id** (int) - table id值。
        - **es_initializer** (EsInitializer) - table_id对应那张大表的EsInitializer对象，推理时可为 ``None``。
        - **embedding_dim** (int) - table_id对应那张大表的插入数据的维度大小。
        - **max_key_num** (int) - 查询的keys的数量。
        - **optimizer_mode** (str) - 优化器类型。默认值为 ``None``。
        - **optimizer_params** (tuple[float]) - 优化器参数。默认值为 ``None``。
        - **es_filter** (CounterFilter) - table_id对应那张大表的特征准入属性。默认值为 ``None``。
        - **es_padding_key** (PaddingParamsOption) - table_id对应那张大表的padding_key属性。默认值为 ``None``。
        - **es_completion_key** (CompletionKeyOption) - table_id对应那张大表的completion_key属性。默认值为 ``None``。

    输入：
        - **keys** (Tensor) - 大表特征keys。
        - **actual_keys_input** (Tensor) - 大表特征keys的所有唯一元素组成的Tensor。
        - **unique_indices** (Tensor) - 大表特征keys每个元素在actual_keys_input中的索引值。
        - **key_count** (Tensor) - actual_keys_input每个元素在大表特征keys的计数值。
