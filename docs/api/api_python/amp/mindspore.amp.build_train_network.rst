mindspore.amp.build_train_network
=================================

.. py:function:: mindspore.amp.build_train_network(network, optimizer, loss_fn=None, level='O0', boost_level='O0', **kwargs)

    构建混合精度训练网络。

    .. note::
        使用 :func:`mindspore.amp.custom_mixed_precision` 或 :func:`mindspore.amp.auto_mixed_precision` 进行精度转换后，不支持再次使用其他接口进行精度转换。
        如果使用 `build_train_network` 来训练转换后的网络，则需要将 `level` 配置为 ``O0`` 以避免重复的精度转换。

    参数：
        - **network** (Cell) - 定义网络结构。
        - **optimizer** (:class:`mindspore.nn.Optimizer`) - 定义优化器，用于更新权重参数。
        - **loss_fn** (Union[None, Cell], 可选) - 定义损失函数。如果为None， `network` 中应该包含损失函数。默认值： ``None`` 。
        - **level** (str, 可选) - 支持['O0', 'O1', 'O2', 'O3', 'auto']。默认值： ``'O0'`` 。

          `level` 的详细配置信息可参考 :func:`mindspore.amp.auto_mixed_precision` 。

          由 `level` 配置的 `keep_batchnorm_fp32`， `cast_model_type` 和 `loss_scale_manager` 可能会被 `kwargs` 里的配置覆盖。

        - **boost_level** (str, 可选) - `mindspore.boost` 中参数 `level` 的选项，设置boost的训练模式级别。支持['O0', 'O1', 'O2']。默认值： ``'O0'`` 。

          - **'O0'** - 不变化。
          - **'O1'** - 开启boost模式，性能提升20%左右，准确率与原始准确率相同。
          - **'O2'** - 开启boost模式，性能提升30%左右，准确率降低小于3%。如果设置了'O1'或'O2'模式，boost相关库将自动生效。

        - **cast_model_type** (mindspore.dtype) - 支持float16，float32。如果设置了该参数，网络将被转化为设置的数据类型，而不会根据设置的level进行转换。
        - **keep_batchnorm_fp32** (bool) - 当网络被设置为float16时，配置为True，则BatchNorm将保持在float32下运行。设置level不会影响该属性。
        - **loss_scale_manager** (Union[None, LossScaleManager]) - 如果不为None，必须是 :class:`mindspore.amp.LossScaleManager` 的子类，用于缩放损失系数(loss scale)。设置level不会影响该属性。

    异常：
        - **ValueError** - 在CPU上，属性 `loss_scale_manager` 不是 `None` 或 :class:`mindspore.amp.FixedLossScaleManager` （其属性 `drop_overflow_update=False` ）。
