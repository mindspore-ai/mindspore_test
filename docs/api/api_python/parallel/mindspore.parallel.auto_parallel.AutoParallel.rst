mindspore.parallel.auto_parallel.AutoParallel
===================================================================================

.. py:class:: mindspore.parallel.auto_parallel.AutoParallel(network, parallel_mode="semi_auto")

    并行配置的基本单元，静态图模式下需封装顶层Cell或函数并指定并行模式。

    .. note::
        使用Model高阶接口进行训练或推理时，传入Model的network必须用AutoParallel进行封装。
        使用函数式训练或推理接口时，必须在最外层进行AutoParallel进行封装，即对AutoParallel进行编译。
        使用函数式训练或推理接口时，暂不支持数据下沉场景。
        

    参数：
        - **network** (Cell或Function，必选) - 待封装的前向网络的顶层Cell或函数，必须是Cell类型或者Python函数。
        - **parallel_mode** (str，可选) - 并行模式。默认为 `semi_auto`，支持：
  
          - ``"semi_auto"``: 半自动化并行模式，支持数据并行、算子级并行、优化器并行和流水线并行场景，默认开启该模式
  
          - ``"sharding_propagation"``: 策略传播搜索模式
  
          - ``"recursive_programming"``: 递归编程搜索模式


    .. py:method:: load_operator_strategy_file(file_path)

        在使用策略传播模式时，设置加载策略JSON文件的路径。

        .. warning::
            - 实验性接口，未来可能变更或移除。
            - 暂不支持加载策略时使用Layout格式。

        参数：
            - **file_path** (dict) - Cell的配置信息，目前用于绑定Cell和数据集。用户也可通过该参数自定义Cell属性。
        
        异常：
            - **TypeError** - 文件路径类型非字符串
            - **KeyError** - 文件路径非绝对路径或非JSON文件后缀结尾
             
    
    .. py:method:: save_operator_strategy_file(file_path)

        在使用策略传播模式时，设置保存策略JSON文件的路径。

        .. warning::
            - 实验性接口，未来可能变更或移除。
            - 暂不支持加载策略时使用Layout格式。

        参数：
            - **file_path** (str) - 绝对路径的JSON文件
  
        异常：
            - **TypeError** - 文件路径类型非字符串
            - **KeyError** - 文件路径非绝对路径或非JSON文件后缀结尾
            

    .. py:method:: dataset_strategy(config)

        设置数据集分片策略。

        参数：
            - **config** (Union[str, tuple(tuple), tuple(Layout)]) - 数据集切分策略配置
  
        异常：
            - **TypeError** - config不是字符串或元组类型
            - **TypeError** - config是元组类型时，元组中的元素不是"元组类型"或者"Layout类型"中的一个
            - **TypeError** - config是元组类型时，且元组中的元素是元组类型时，子元组的元素类型不是"int"
            - **ValueError** - 输入config为空
            - **ValueError** - config为字符串类型时，取值不是"full_batch"或"data_parallel"中的一个


    .. py:method:: print_local_norm()

        在自动/半自动并行场景下，开启后打印local norm值。


    .. py:method:: dump_local_norm(file_path)

        在自动/半自动并行场景下，指定local norm值的保存路径。

        参数：
            - **file_path** (str) - 保存路径，默认值为 `""`
        异常：
            - **TypeError** - 文件路径类型非字符串


    .. py:method:: enable_device_local_norm()

        在自动/半自动并行场景下，开启后打印device local norm值。


    .. py:method:: no_init_parameters_in_compile()

        开启后，限制并行编译过程中的模型权重参数初始化。

        .. warning::
            - 实验性接口，未来可能变更或移除。