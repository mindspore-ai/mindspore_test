mindspore
=========

Data Presentation
------------------

Tensor
^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Tensor
    mindspore.tensor
    mindspore.COOTensor
    mindspore.CSRTensor
    mindspore.RowTensor
    mindspore.SparseTensor

Parameter
^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Parameter
    mindspore.ParameterTuple

DataType
^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dtype
    mindspore.dtype_to_nptype
    mindspore.dtype_to_pytype
    mindspore.pytype_to_dtype
    mindspore.get_py_obj_dtype
    mindspore.QuantDtype
    mindspore.common.np_dtype

Context
--------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.set_context
    mindspore.get_context
    mindspore.set_auto_parallel_context
    mindspore.get_auto_parallel_context
    mindspore.reset_auto_parallel_context
    mindspore.ParallelMode
    mindspore.set_ps_context
    mindspore.get_ps_context
    mindspore.reset_ps_context
    mindspore.set_algo_parameters
    mindspore.get_algo_parameters
    mindspore.reset_algo_parameters
    mindspore.set_offload_context
    mindspore.get_offload_context

Seed
----

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.set_seed
    mindspore.get_seed

Random Number Generator
-----------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.get_rng_state
    mindspore.Generator
    mindspore.initial_seed
    mindspore.manual_seed
    mindspore.seed
    mindspore.set_rng_state

Serialization
-------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.async_ckpt_thread_status
    mindspore.build_searched_strategy
    mindspore.check_checkpoint
    mindspore.ckpt_to_safetensors
    mindspore.convert_model
    mindspore.export
    mindspore.get_ckpt_path_with_strategy
    mindspore.load
    mindspore.load_checkpoint
    mindspore.load_checkpoint_async
    mindspore.load_distributed_checkpoint
    mindspore.load_mindir
    mindspore.load_param_into_net
    mindspore.load_segmented_checkpoints
    mindspore.merge_pipeline_strategys
    mindspore.merge_sliced_parameter
    mindspore.obfuscate_model
    mindspore.parse_print
    mindspore.rank_list_for_transform
    mindspore.restore_group_info_list
    mindspore.safetensors_to_ckpt
    mindspore.save_checkpoint
    mindspore.save_mindir
    mindspore.transform_checkpoint_by_rank
    mindspore.transform_checkpoints

Automatic Differentiation
---------------------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.grad
    mindspore.value_and_grad
    mindspore.get_grad
    mindspore.jacfwd
    mindspore.jacrev
    mindspore.jvp
    mindspore.vjp

Parallel Optimization
-----------------------

Automatic Vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.vmap

Parallel
^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Layout
    mindspore.parameter_broadcast
    mindspore.recompute
    mindspore.reshard
    mindspore.shard
    mindspore.sync_pipeline_shared_parameters
    
JIT
---

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.JitConfig
    mindspore.jit
    mindspore.jit_class
    mindspore.ms_class
    mindspore.ms_function
    mindspore.ms_memory_recycle
    mindspore.mutable
    mindspore.constexpr
    mindspore.lazy_inline
    mindspore.no_inline

Tool
-----

Dataset Helper
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.DatasetHelper
    mindspore.Symbol
    mindspore.connect_network_with_dataset
    mindspore.data_sink

Debugging and Tuning
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Profiler
    mindspore.profiler.DynamicProfilerMonitor
    mindspore.SummaryCollector
    mindspore.SummaryLandscape
    mindspore.SummaryRecord
    mindspore.set_dump 

Log
^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.get_level
    mindspore.get_log_config

Installation Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.run_check
