mindspore.parallel
=============================

mindspore.parallel provides a comprehensive set of automatic parallel interfaces, including basic parallel configuration units, model serialization, parallel partitioning strategy interfaces, and more.

The module import method is as follows:

.. code-block::

    from mindspore import parallel

Parallel Configuration Units
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.auto_parallel.AutoParallel
    mindspore.parallel.nn.GradAccumulation
    mindspore.parallel.nn.MicroBatchInterleaved
    mindspore.parallel.nn.Pipeline
    mindspore.parallel.nn.PipelineGradReducer


Model Serialization
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.parallel.convert_checkpoints
    mindspore.parallel.convert_checkpoint_by_rank
    mindspore.parallel.load_distributed_checkpoint
    mindspore.parallel.load_segmented_checkpoints
    mindspore.parallel.rank_list_for_convert


Parallel Partitioning Strategy
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.build_searched_strategy
    mindspore.parallel.function.reshard
    mindspore.parallel.Layout
    mindspore.parallel.merge_pipeline_strategys
    mindspore.parallel.set_op_strategy_config
    mindspore.parallel.shard


others
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.parameter_broadcast
    mindspore.parallel.restore_group_info_list
    mindspore.parallel.sync_pipeline_shared_parameters