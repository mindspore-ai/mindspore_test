mindspore.dataset
=================

MindSpore Dataset is a high-performance data engine module specifically designed within the MindSpore framework,
dedicated to providing efficient, flexible, and user-friendly data loading and preprocessing solutions for deep learning tasks.
It supports multiple data formats (such as MindRecord, TFRecord, etc.) and includes a rich set of built-in public dataset interfaces,
enabling users to quickly construct data pipelines. With MindSpore Dataset, users can effortlessly perform data reading, transformation,
and augmentation, meeting the processing needs of various data types such as images, text, and audio.

Additionally, MindSpore Dataset offers powerful data transformation capabilities, supporting a variety of data augmentation operations
(e.g., cropping, rotation, normalization, etc.), which can effectively enhance the generalization ability of models.
By leveraging the efficient MindRecord data storage format, users can further optimize data reading performance, significantly accelerating large-scale data training tasks.
The design of MindSpore Dataset balances flexibility and performance, supporting both single-machine and distributed training scenarios.
It seamlessly integrates into the model development and training workflows of MindSpore, providing users with end-to-end efficient support from data preprocessing to model training.


- Dataset Loading( `mindspore.dataset <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html>`_ ),
  this module provides multiple data loading methods to help users load datasets into MindSpore.


- Data Argumentation( `mindspore.dataset.transforms <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html>`_ ),
  this module provides common data transformations in the fields of image, text and audio, and also supports
  customized data transformations to help users apply data argumentation online.


- MindRecord format( `mindspore.mindrecord <https://www.mindspore.cn/docs/en/master/api_python/mindspore.mindrecord.html>`_ ),
  this module provides an efficient data format that helps users to easily convert data sources to a standard format
  and supports high-speed reads during training.