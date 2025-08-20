mindspore.dataset.dataloader
============================

This module provides iterators for loading datasets. It supports loading both Map Style and Iterable Style datasets, and offers multi-process concurrent loading.

Data Loader
---------------

.. msautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.DataLoader

Datasets
------------

.. msautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.Dataset
    mindspore.dataset.dataloader.IterableDataset
    mindspore.dataset.dataloader.TensorDataset

Samplers
------------

.. msautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.Sampler
    mindspore.dataset.dataloader.SequentialSampler
    mindspore.dataset.dataloader.RandomSampler
    mindspore.dataset.dataloader.BatchSampler
    mindspore.dataset.dataloader.DistributedSampler

Collate Functions
---------------------

.. msautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.default_collate
    mindspore.dataset.dataloader.default_convert
    mindspore.dataset.dataloader._utils.collate.collate

Utilities
-------------

.. msautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.get_worker_info
