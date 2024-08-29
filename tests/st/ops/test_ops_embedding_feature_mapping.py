# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, nn, context
from tests.mark_utils import arg_mark


class EmbeddingFeatueMappingV2Net(nn.Cell):
    """
    EmbeddingFeatueMappingV2Net
    """

    def construct(self, table_name, feature_id, table_total_size, table_actual_size):
        return ops.auto_generate.embedding_feature_mapping_v2(table_name, feature_id,
                                                              table_total_size, table_actual_size)


class EmbeddingFeatueMappingExportNet(nn.Cell):
    """
    EmbeddingFeatueMappingExportNet
    """

    def __init__(self, table_name):
        super(EmbeddingFeatueMappingExportNet, self).__init__()
        self.table_name = Tensor(np.array(table_name))
        self.len = len(table_name)
        self.table_name_list = [Tensor(np.array([name])) for name in table_name]

    def construct(self, file_path, values, embedding_dim):
        feature_id = []
        offset_id = []
        for i in range(self.len):
            cur_table_name = self.table_name_list[i]
            cur_feature_size = ops.auto_generate.embedding_feature_mapping_table_size(cur_table_name)
            f, o = ops.auto_generate.embedding_feature_mapping_find(cur_table_name, cur_feature_size, 1)
            feature_id.append(f)
            offset_id.append(o)
        out = ops.auto_generate.embedding_feature_mapping_export(file_path, self.table_name,
                                                                 values, embedding_dim,
                                                                 feature_id, offset_id)
        return out


class EmbeddingFeatueMappingInsertNet(nn.Cell):
    """
    EmbeddingFeatueMappingInsertNet
    """

    def __init__(self, table_name):
        super(EmbeddingFeatueMappingInsertNet, self).__init__()
        self.table_name = Tensor(np.array(table_name))
        self.len = len(table_name)
        self.table_name_list = [Tensor(np.array([name])) for name in table_name]

    def construct(self, file_path, embedding_dim, only_offset_flag=True):
        feature_id = []
        offset_id = []
        for i in range(self.len):
            cur_table_name = self.table_name_list[i]
            cur_embedding_dim = embedding_dim[i:i + 1]
            cur_feature_size = ops.auto_generate.embedding_feature_mapping_file_size(file_path,
                                                                                     cur_table_name,
                                                                                     cur_embedding_dim,
                                                                                     only_offset_flag)
            f, o = ops.auto_generate.embedding_feature_mapping_import(file_path, cur_table_name,
                                                                      cur_feature_size, cur_embedding_dim,
                                                                      only_offset_flag, 1)
            feature_id.append(f)
            offset_id.append(o)
        out = ops.auto_generate.embedding_feature_mapping_insert(self.table_name, 1, feature_id, offset_id)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_embedding_feature_mapping_test():
    """
    Feature: Ops
    Description: test FeatureMapping
    Expectation: expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": 'O2'})
    os.environ['MS_DISABLE_REF_MODE'] = "1"

    embedding_feature_mapping_v2_forward_func = EmbeddingFeatueMappingV2Net()
    feature_id = Tensor([1], ms.int64)
    table_total_size = (2,)
    table_actual_size = (2,)
    offset_id0 = embedding_feature_mapping_v2_forward_func("0001", feature_id,
                                                           table_total_size, table_actual_size)
    offset_id1 = embedding_feature_mapping_v2_forward_func("0002", feature_id,
                                                           table_total_size, table_actual_size)
    print(f"embedding_feature_mapping_v2_forward_func offset_id: {offset_id0}, {offset_id1}")

    table_name = ["0001", "0002"]
    embedding_dim = [10, 20]
    file_path = os.path.join(os.getcwd(), "embedding")
    values = Tensor(0, ms.float32)
    for i in range(len(table_name)):
        cur_table_name = table_name[i:i + 1]
        cur_embedding_dim = embedding_dim[i:i + 1]
        embedding_feature_mapping_export_forward_func = EmbeddingFeatueMappingExportNet(cur_table_name)
        embedding_feature_mapping_export_forward_func(file_path, values, cur_embedding_dim)

        embedding_feature_mapping_insert_forward_func = EmbeddingFeatueMappingInsertNet(cur_table_name)
        embedding_feature_mapping_insert_forward_func(file_path, cur_embedding_dim)
