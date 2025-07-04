mindspore.dataset.RenderedSST2Dataset
=====================================

.. py:class:: mindspore.dataset.RenderedSST2Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    RenderedSST2（Rendered Stanford Sentiment Treebank v2）数据集。

    生成的数据集有两列 `[image, label]`。`image` 列的数据类型为uint8。`label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 ``'train'`` 、 ``'val'`` 、 ``'test'`` 或 ``'all'`` 。默认值： ``None`` ，读取全部样本图片。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数(8)，也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` ，下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``False`` ，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` ，下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `usage` 参数取值不为 ``'train'`` 、 ``'val'`` 、 ``'test'`` 或 ``'all'`` 。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `shard_id` 参数值错误，小于0或者大于等于 `num_shards` 。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于RenderedSST2数据集：**

    Rendered SST2是一个图像分类数据集，它是由SST2数据集中的数据生成的。数据集被分割成三份，每一份包含有两类（positive和negative）：
    在train这一份下共有6920张图像（3610张positive，3310张negative），在validation这一份下共有872张图像（444张positive，428张negative），
    在test这一份下共有1821张图像（909张positive，912张negative）。

    以下为原始RenderedSST2数据集的结构，您可以将数据集文件解压得到如下的文件结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── rendered_sst2_dataset_directory
             ├── train
             │    ├── negative
             │    │    ├── 0001.jpg
             │    │    ├── 0002.jpg
             │    │    ...
             │    └── positive
             │         ├── 0001.jpg
             │         ├── 0002.jpg
             │         ...
             ├── test
             │    ├── negative
             │    │    ├── 0001.jpg
             │    │    ├── 0002.jpg
             │    │    ...
             │    └── positive
             │         ├── 0001.jpg
             │         ├── 0002.jpg
             │         ...
             └── valid
                  ├── negative
                  │    ├── 0001.jpg
                  │    ├── 0002.jpg
                  │    ...
                  └── positive
                       ├── 0001.jpg
                       ├── 0002.jpg
                       ...

    **引用：**

    .. code-block::

        @inproceedings{socher-etal-2013-recursive,
            title     = {Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
            author    = {Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning,
                          Christopher D. and Ng, Andrew and Potts, Christopher},
            booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
            month     = oct,
            year      = {2013},
            address   = {Seattle, Washington, USA},
            publisher = {Association for Computational Linguistics},
            url       = {https://www.aclweb.org/anthology/D13-1170},
            pages     = {1631--1642},
        }


.. include:: mindspore.dataset.api_list_vision.rst
