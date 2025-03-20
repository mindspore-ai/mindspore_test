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


    .. py:method:: comm_fusion(config)

        用于设置并行通信算子的融合配置。
        
        参数：
            - **config** (dict) - 输入格式为{"通信类型": {"mode":str, "config": None int 或者 list}}, 每种通信算子的融合配置有两个键："mode"和"config"。支持以下通信类型的融合类型和配置：
                  
              - openstate：是否开启通信融合功能。通过 ``True`` 或 ``False`` 来开启或关闭通信融合功能。默认值： ``True``，开启通信融合功能。
              - allreduce：进行AllReduce算子的通信融合。"mode"包含"auto"、"size"和"index"。在"auto"模式下，融合梯度变量的大小，默认值阈值为"64"MB，"config"对应的值为None。在"size"模式下，需要用户在config的字典中指定梯度大小阈值，这个值必须大于"0"MB。在"mode"为"index"时，它与"all_reduce_fusion_config"相同，用户需要给"config"传入一个列表，里面每个值表示梯度的索引。
              - allgather：进行AllGather算子的通信融合。"mode"包含"auto"、"size"。在 'auto' 模式下，AllGather融合由梯度值决定，其默认融合配置阈值为 '64' MB。在 'size' 模式下，手动设置AllGather算子融合的梯度阈值，并且其融合阈值必须大于 '0' MB。
              - reducescatter：进行ReduceScatter算子的通信融合。"mode"包含"auto"、"size"，"auto" 和 "size"模式的配置方式与allgather相同。
            
        异常：
            - **TypeError** - 配置项非字典类型。


    .. py:method:: disable_strategy_file_only_for_trainable_params()

        默认情况，MindSpore仅保存/加载可训练参数的策略信息，调用此接口后，可支持加载和保存模型的非可训练参数。


    .. py:method:: enable_fp32_communication()

        开启之后，通信期间Reduce类算子（AllReduce、ReduceScatter）强制使用fp32数据类型进行通信。


    .. py:method:: load_param_strategy_file(file_path)

        设置加载并行策略checkpoint的路径，默认仅加载可训练参数的策略信息。

        参数：
            - **file_path** (str) - 加载路径。

        异常：
            - **TypeError** - 文件路径类型非字符串。


    .. py:method:: save_param_strategy_file(file_path)

        设置保存并行策略checkpoint的路径，默认仅保存可训练参数的策略信息。

        参数：
            - **file_path** (str) - 保存路径。

        异常：
            - **TypeError** - 文件路径类型非字符串。


    .. py:method:: set_group_ckpt_save_file(file_path)

        在自动/半自动并行场景下，指定图编译过程中所创建group的保存路径。

        参数：         
            - **file_path** (str) - 保存路径。

        异常：
            - **TypeError** - 文件路径类型非字符串。


    .. py:method:: transformer_opt(file_path)

        并行加速配置文件，配置项可以参考 parallel_speed_up.json。当设置为None时，表示不启用。

        参数：
            - **file_path** (Union[str, None]): 并行加速配置文件，配置项可以参考 `parallel_speed_up.json <https://gitee.com/mindspore/mindspore/blob/master/config/parallel_speed_up.json>`_ 。
              当设置为None时，表示不启用。

              - **recomputation_communication_overlap** (bool): 为 ``True`` 时表示开启反向重计算和通信掩盖。默认值： ``False`` 。
              - **grad_matmul_communication_overlap** (bool): 为 ``True`` 时表示开启反向Matmul和通信掩盖。默认值： ``False`` 。
              - **grad_fa_allgather_overlap** (bool):为 ``True`` 时表示在序列并行和开启FlashAttentionScoreGrad算子时，开启重计算以掩盖重复的AllGather。默认值： ``False`` 。
              - **enable_communication_fusion** (bool): 为 ``True`` 时表示开启通信融合进行通信算子task数量优化。默认值： ``False`` 。
              - **grad_computation_allreduce_overlap** (bool): 为 ``True`` 时表示开启梯度dx计算与数据并行梯度通信的掩盖，暂时不支持 `O2 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html>`_ 编译模式下开启。注意在数据并行梯度通信和计算掩盖良好的情况下，开启该选项后性能不一定有提升，请根据实际场景确定是否开启。默认值： ``False`` 。
              - **computation_allgather_overlap** (bool): 为 ``True`` 时表示开启正向计算与优化器并行的AllGather通信的掩盖，暂时不支持 `O2 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html>`_ 编译模式下开启。注意在权重聚合通信和计算掩盖良好的情况下，开启该选项后性能不一定有提升，请根据实际场景确定是否开启。默认值： ``False`` 。
              - **enable_concat_eliminate_opt** (bool): 为 ``True`` 时表示开启Concat消除优化，当前在开启细粒度双副本优化时有收益。默认值： ``False`` 。
              - **enable_begin_end_inline_opt** (bool): 为 ``True`` 时表示开启首尾micro_batch子图的内联，用于半自动并行子图模式，流水线并行场景，一般需要和其他通信计算掩盖优化一起使用。默认值： ``False`` 。
              - **computation_communication_fusion_level** (int): 控制通算融合的级别。默认值： ``0`` 。注：此功能需要配套Ascend Training Solution 24.0.RC2以上版本使用。

                - 0: 不启用通算融合。

                - 1: 仅对前向节点使能通算融合。

                - 2: 仅对反向节点使能通算融合。

                - 3: 对所有节点使能通算融合。

              - **dataset_broadcast_opt_level** (int): 数据集读取的优化级别， 目前只支持O0/O1模式，O2模式下不生效。默认值： ``0`` 。

                - 0: 不启用数据集读取优化。

                - 1: 优化流水线并行中，Stage间的数据读取。

                - 2: 优化模型并行维度数据的读取。

                - 3: 同时优化场景1和2。

              - **allreduce_and_biasadd_swap** (bool): 为 ``True`` 时表示开启matmul-add结构下，通信算子与Add算子执行顺序互换。当前仅支持bias为一维的情况。默认值： ``False`` 。
              - **enable_allreduce_slice_to_reducescatter** (bool): 为 ``True`` 时，表示开启AllReduce优化。在batchmatmul模型并行引入AllReduce的场景中，如果后续节点是配置了模型并行的StridedSlice算子，在已识别可优化的模式中，将AllReduce优化为ReduceScatter。典型的用在开启了groupwise alltoall的MoE模块。默认值： ``False`` 。
              - **enable_interleave_split_concat_branch** (bool): 为 ``True`` 时，表示针对带enable_interleave属性的Split和Concat算子形成的分支，开启通信计算并行优化。典型的使用场景为MoE模块并行场景，对输入数据进行split后，对各切片数据进行MoE模块运算，再对分支结果进行Concat，开启后各分支的MoE模块进行通信计算并行优化。默认值： ``False`` 。
              - **enable_interleave_parallel_branch** (bool): 为 ``True`` 时，表示针对可并行的分支，如果分支汇聚点带parallel_branch属性，开启通信计算并行优化。典型的使用场景为MoE模块带路由专家和共享专家分支的并行场景，开启后并行分支进行通信计算并行优化。默认值： ``False`` 。
