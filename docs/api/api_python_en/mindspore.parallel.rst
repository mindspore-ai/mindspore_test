mindspore.parallel
=============================

mindspore.parallel provides a comprehensive set of automatic parallel interfaces, including basic parallel configuration units, model serialization, parallel partitioning strategy interfaces, and more.

The module import method is as follows:

.. code-block::

    from mindspore import parallel

Parallel Configuration Units
----------------------------------------------------------------

.. msplatwarnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.auto_parallel.AutoParallel
    mindspore.parallel.nn.Pipeline
    mindspore.parallel.nn.PipelineGradReducer
    mindspore.parallel.nn.GradAccumulation
    mindspore.parallel.nn.MicroBatchInterleaved


Model Serialization
----------------------------------------------------------------

.. msplatwarnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.load_distributed_checkpoint
    mindspore.parallel.convert_checkpoints
    mindspore.parallel.rank_list_for_convert
    mindspore.parallel.convert_checkpoint_by_rank
    mindspore.parallel.load_segmented_checkpoints


Parallel Partitioning Strategy
----------------------------------------------------------------

.. msplatwarnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.build_searched_strategy
    mindspore.parallel.function.reshard
    mindspore.parallel.shard
    mindspore.parallel.layout


others
----------------------------------------------------------------

.. msplatwarnautosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.restore_group_info_list
    mindspore.parallel.parameter_broadcast
    mindspore.parallel.sync_pipeline_shared_parameters