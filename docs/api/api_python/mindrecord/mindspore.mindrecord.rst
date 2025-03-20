MindRecord是MindSpore提供的高效数据存储/读取模块，此模块提供了一些方法帮助用户将不同公开数据集转换为MindRecord格式，
也提供了一些方法对MindRecord数据文件进行读取、写入、检索等。

.. image:: data_conversion_concept.png

MindSpore格式数据可以更加方便地保存和加载数据，其目标是归一化用户的数据集，并针对不同数据场景进行了性能优化。
使用MindRecord数据格式可以减少磁盘IO、网络IO开销，从而获得更好的数据加载体验。

用户可以使用 `mindspore.mindrecord.FileWriter` 生成MindRecord格式数据文件，并使用 `mindspore.dataset.MindDataset <https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html>`_ 加载MindRecord格式数据集。

用户还可以将其他格式数据集转换为MindRecord格式数据集，详见 `MindRecord格式转换 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/record.html>`_ 。
同时，MindRecord还支持配置文件加密、解密和完整性校验，以保证MindRecord格式数据集的安全。
