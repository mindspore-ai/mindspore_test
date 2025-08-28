mindspore.parallel
=============================

mindspore.parallel provides a large number of interfaces for automatic parallelization, including parallel base configuration, model loading and transformation, and functional parallel slicing.

The module import method is as follows:

.. code-block::

    from mindspore import parallel

Parallel Base Configuration
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.auto_parallel.AutoParallel
    mindspore.parallel.distributed.DistributedDataParallel
    mindspore.parallel.nn.GradAccumulation
    mindspore.parallel.nn.MicroBatchInterleaved
    mindspore.parallel.nn.Pipeline
    mindspore.parallel.nn.PipelineGradReducer


Model Loading and Transformation
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
    mindspore.parallel.unified_safetensors


Functional Parallel Slicing
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.function.reshard
    mindspore.parallel.Layout
    mindspore.parallel.shard


Others
----------------------------------------------------------------

.. autosummary::
    :toctree: parallel
    :nosignatures:
    :template: classtemplate.rst

    mindspore.parallel.build_searched_strategy
    mindspore.parallel.merge_pipeline_strategys
    mindspore.parallel.parameter_broadcast
    mindspore.parallel.restore_group_info_list
    mindspore.parallel.sync_pipeline_shared_parameters
    mindspore.parallel.strategy.enable_save_strategy_online
    mindspore.parallel.strategy.get_strategy_metadata
    mindspore.parallel.strategy.get_current_strategy_metadata
    mindspore.parallel.strategy.clear_strategy_metadata