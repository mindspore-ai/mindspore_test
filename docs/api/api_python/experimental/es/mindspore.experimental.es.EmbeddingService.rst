mindspore.experimental.es.EmbeddingService
==============================================

.. py:class:: mindspore.experimental.es.EmbeddingService()

    目前ES(EmbeddingService)特性只能创建一个EmbeddingService对象的实例，可支持大小表Embedding的模型训练和推理，为训练和推理提供统一的Embedding管理，存储和计算的能力。
    其中大表指vocab_size超过10万的数据量，推荐用户存储在PS(Parameter Server)上的表；
    小表指vocab_size小于10万的数据量，推荐用户存储在device上的表。

    .. warning::
        这是一个实验性的EmbeddingService接口。

    .. note::此接口需要先调用 `mindspore.communication.init()` 完成动态组网后才能生效。

    异常：
        - **ValueError** - 实例化对象的时候，未设置ESCLUSTER_CONFIG_PATH环境变量会报错。
        - **ValueError** - ESCLUSTER_CONFIG_PATH路径对应的配置文件中，每个server参数服务器总数超过4。
        - **ValueError** - ESCLUSTER_CONFIG_PATH路径对应的配置文件中，参数服务器总数超过4。

    .. py:method:: embedding_init(embedding_dim, init_vocabulary_size, name, allow_merge=False, embedding_type="PS", ev_option=None, initializer=Uniform(scale=0.01), max_feature_count=None, mode="train", multihot_lens=None, optimizer=None, optimizer_param=None)

        大小表的初始化接口。

        参数：
            - **name** (str) - 表名。
            - **init_vocabulary_size** (int) - 初始化表的大小。
            - **embedding_dim** (int) - 表中插入数据的维度大小。
            - **max_feature_count** (int) - 每次查询的keys的数量。默认值为 ``None``。
            - **initializer** (Initializer) - 表的初始化策略，默认值为 ``Uniform``。
            - **embedding_type** (str) - 生成embedding表的类型，可配置参数["PS", "data_parallel"]， ``"PS"``表示初始化大表，``"data_parallel"``表示初始化小表。默认值为 ``"PS"``。
            - **ev_option** (EmbeddingVariableOption) - 大表的一些属性，是embedding_variable_option函数的返回值，为EmbeddingVariableOption对象。默认值为 ``None``。
            - **multihot_lens** (int) - 小表的属性，只有allow_merge使能之后才用，当前不支持。默认值为 ``None``。
            - **optimizer** (str) - 训练场景下大表优化器的类型，每张大表不能共用优化器，当前只支持``"Adam"``、``"Ftrl"``、``"SGD"``和``"RMSProp"``。默认值为 ``None``。
            - **allow_merge** (bool) - 小表的属性，表示是否进行小表合并，当前不支持。默认值为 ``False``。
            - **optimizer_param** (float) - 用户配置的优化器参数：initial_accumulator_value，表示moment累加器的初始值。默认值为 ``None``。
            - **mode** (str) - 网络运行模式，可配置参数["train", "predict", "export"]， ``"train"``表示训练模式，``"predict"``表示推理模式，``"export"``表示模型导出模式。默认值为 ``"train"``。

        返回：
            - 如果初始化小表，返回小表的dict信息。
            - 如果初始化大表，返回EmbeddingServiceOut对象，包含5个参数：table_id_dict、 es_initializer、 es_counter_filter、 es_padding_keys、 es_completion_keys。

              - table_id_dict(dict)，大表名和table_id的键值对。
              - es_initializer(dict)，table_id和大表属性EsInitializer对象的键值对。
              - es_counter_filter(dict)，table_id和准入条件的键值对。
              - es_padding_keys(dict)，table_id和padding key条件的键值对。
              - es_completion_keys(dict)，table_id和completion ket的键值对。

        异常：
            - **ValueError** - 如果未设置 `name` ， `init_vocabulary_size` ， `embedding_dim`， `max_feature_count` 。
            - **ValueError** - 如果 `name` ， `init_vocabulary_size` ， `embedding_dim`， `max_feature_count` 类型不匹配。
            - **ValueError** - 如果 `init_vocabulary_size` ， `embedding_dim` ， `max_feature_count` 值小于等于0，或 `init_vocabulary_size` 值大于2147483647。
            - **ValueError** - 如果大表个数超过1024。
            - **ValueError** - 如果 `optimizer` 不属于["adam", "adagrad", "adamw", "ftrl", "sgd", "rmsprop"]。
            - **TypeError** - 如果 `initializer` 不是EsInitializer对象，或者不属于["TruncatedNormal", "Uniform", "Constant"]。

    .. py:method:: padding_param(padding_key, mask=True, mask_zero=False)

        为每张大表初始化padding key策略。

        参数：
            - **padding_key** (int) - padding的值，必须是真实合法的hash key。
            - **mask** (bool) - 是否更新padding key，设置为false则不更新。默认值为 ``True``。
            - **mask_zero** (bool) - 是否更新key==0的值。默认值为 ``False``。

        返回：
            PaddingParamsOption对象。

        异常：
            - **TypeError** - 如果padding_key入参类型不是int。
            - **TypeError** - 如果mask入参类型不是bool。

    .. py:method:: completion_key(completion_key, mask=True)

        为每张大表初始化completion key策略。

        参数：
            - **completion_key** (int) - completion key取值。
            - **mask** (bool) - 是否更新completion key，设置为false则不更新。默认值为 ``True``。

        返回：
            CompletionKeyOption对象。

        异常：
            - **TypeError** - 如果completion_key入参类型不是int。
            - **TypeError** - 如果mask入参类型不是bool。

    .. py:method:: counter_filter(filter_freq, default_key=None, default_value=None)

        为每张大表设置准入策略。

        .. note::
            此功能仅支持训练模式。如果训练时使用了特征准入，而后执行推理，对于某些key查询不到的情况，当前支持使用default_vlaue进行设置。

        参数：
            - **filter_freq** (int) - 特征准入的频率阈值。
            - **default_key** (int) - 出现次数未达到阈值的key特征，在查询是返回default_key对应的特征值。默认值为 ``None``。
            - **default_value** (int/float) - 出现次数未达到阈值的key特征，在查询时返回embedding_dim长度的default_value作为其特征值。默认值为 ``None``。

        返回：
            CounterFilter对象。

        异常：
            - **TypeError** - 如果 `filter_freq` 类型不是int。
            - **ValueError** - 如果 `filter_freq` 小于0。
            - **ValueError** - 如果 `default_key` 和 `default_value` 都是None。
            - **ValueError** - 如果 `default_key` 和 `default_value` 都不是None。
            - **TypeError** - 如果 `default_key` 是None，且 `default_value` 不是int或float。
            - **TypeError** - 如果 `default_value` 是None，且 `default_key` 不是int。

    .. py:method:: evict_option(steps_to_live)

        为每张大表设置被动淘汰策略。

        参数：
            - **steps_to_live** (int) - 特征key设置的淘汰存活步数。

        返回：
            EvictOption对象。

        异常：
            - **TypeError** - 如果 `steps_to_live` 类型不是int。
            - **ValueError** - 如果 `steps_to_live` 不大于0。


    .. py:method:: embedding_variable_option(filter_option=None, padding_option=None, evict_option=None, completion_option=None, storage_option=None, feature_freezing_option=None, communication_option=None)

        每个大表的所有属性配置合集。

        参数：
            - **filter_option** (CounterFilter) - filter属性。默认值为 ``None``。
            - **padding_option** (PaddingParamsOption) - padding key属性。默认值为 ``None``。
            - **evict_option** (EvictOption) - evict属性。默认值为 ``None``。
            - **completion_option** (CompletionKeyOption) - completion key属性。默认值为 ``None``。
            - **storage_option** (None) - 预留属性，当前不支持。默认值为 ``None``。
            - **feature_freezing_option** (None) - 预留属性，当前不支持。默认值为 ``None``。
            - **communication_option** (None) - 预留属性，当前不支持。默认值为 ``None``。

        返回：
            EmbeddingVariableOption对象，作为embedding_init的ev_option入参。

        异常：
            - **TypeError** - 如果filter_option不是None且类型不是CounterFilter。
            - **TypeError** - 如果padding_option不是None且类型不是PaddingParamsOption。
            - **TypeError** - 如果completion_option不是None且类型不是CompletionKeyOption。
            - **TypeError** - 如果evict_option不是None且类型不是EvictOption。

    .. py:method:: embedding_ckpt_export(file_path， trainable_var)

        导出每张大表的embedding table和优化器参数，及小表的embedding tale。

        .. note::
            此功能只支持rank 0执行。针对大表，在导出之前需要先调用embedding_variable_option接口为每张表设置对应的被动淘汰条件。

        参数：
            - **file_path** (str) - 存放ckpt的地址，最后一个字符不能为 ``"/"``。
            - **trainable_var** (list[parameter]) - 存放所有小表的parameter信息。

        返回：
            EmbeddingComputeVarExport算子的返回值及小表的导出结果。

    .. py:method:: embedding_table_export(file_path， trainable_var)

        导出每张大表及小表的embedding table信息。

        .. note::
            此功能只支持rank 0执行。

        参数：
            - **file_path** (str) - 存放table表的地址，最后一个字符不能为 ``"/"``。
            - **trainable_var** (list[parameter]) - 存放所有小表的parameter信息。

        返回：
            EmbeddingTableExport算子的返回值，及小表的embedding tale。

    .. py:method:: incremental_embedding_table_export(file_path)

        导出每张大表进行embedding table的增量导出。

        .. note::
            此功能只支持rank 0执行。

        参数：
            - **file_path** (str) - 增量导出table表的存放地址，最后一个字符不能为 ``"/"``。

        返回：
            EmbeddingTableExport算子的返回值。

    .. py:method:: embedding_ckpt_import(file_path)

        导入file path下所有的embedding文件和ckpt文件。

        参数：
            - **file_path** (str) - 存放ckpt的地址，最后一个字符不能为 ``"/"``。

        返回：
            EmbeddingComputeVarImport算子的返回值及小表导入结果。

    .. py:method:: embedding_table_import(file_path)

        导入file path下所有的embedding文件。

        参数：
            - **file_path** (str) - 存放table表的地址，最后一个字符不能为 ``"/"``。

        返回：
            EmbeddingTableImport算子的返回值及小表导入结果。

    .. py:method:: embedding_evict(steps_to_live)

        为所有大表使能主动淘汰。

        参数：
            - **steps_to_live** (int) - 特征key设置的淘汰存活步数。

        返回：
            EmbeddingTableEvict算子的返回值。

        异常：
            - **TypeError** - 如果 `steps_to_live` 类型不是int。
            - **ValueError** - 如果 `steps_to_live` 不大于0。

    .. py:method:: init_table()

        为小表初始化相应的参数。

        返回：
            所有已初始化的小表信息，类型：dict，key为每张小表的名字，value为初始化的parameter。