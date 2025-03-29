mindspore.parallel
==========================================================

mindspore.parallel提供了大量的自动并行接口，包括并行配置基本单元、模型保存与加载、并行切分策略接口等。

模块导入方法如下：

.. code-block::

    from mindspore import parallel

并行配置基本单元
----------------------------------------------------------------

.. mscnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.auto_parallel.AutoParallel
    mindspore.parallel.nn.Pipeline
    mindspore.parallel.nn.PipelineGradReducer
    mindspore.parallel.nn.GradAccumulation
    mindspore.parallel.nn.MicroBatchInterleaved


模型保存与加载
----------------------------------------------------------------

.. mscnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.load_distributed_checkpoint
    mindspore.parallel.convert_checkpoints
    mindspore.parallel.rank_list_for_convert
    mindspore.parallel.convert_checkpoint_by_rank
    mindspore.parallel.load_segmented_checkpoints


并行切分策略接口
----------------------------------------------------------------

.. mscnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.build_searched_strategy
    mindspore.parallel.function.reshard
    mindspore.parallel.shard
    mindspore.parallel.Layout


其他
----------------------------------------------------------------

.. mscnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.restore_group_info_list
    mindspore.parallel.parameter_broadcast
    mindspore.parallel.sync_pipeline_shared_parameters