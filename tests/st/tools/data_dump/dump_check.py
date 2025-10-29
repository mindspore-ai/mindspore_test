# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Check the dump result functionality
"""
import os
from mindspore import log as logger

FILE_RWGR_MODE = '640'
FILE_RW_MODE = '600'

WORK_DIR = os.path.realpath('./')
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))

STATISTIC_HEAD = "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape"
HEAD_DICT = {"max": "Max Value", "min": "Min Value", "avg": "Avg Value", "count": "Count",
             "negative zero count": "Negative Zero Count", "positive zero count": "Positive Zero Count",
             "nan count": "NaN Count", "negative inf count": "Negative Inf Count",
             "positive inf count": "Positive Inf Count",
             "zero count": "Zero Count", "md5": "MD5", "l2norm": "L2Norm Value", "hash:md5": "MD5", "hash": "SHA1",
             "hash:sha1": "SHA1"}


IGNORE_OPTYPE_LIST = ["ge:Const", "ge:NoOp", "ge:Send", "ge:Recv", "ge:Variable", "ge:NetOutput", "ge:ConcatV2D",
                      "ge:If", "ge:RefData", "ge:StreamActive", "ge:MemcpyAsync", "ge:Assign", "ge:StreamSwitch",
                      "ge:ConcatD", "ge:SplitVD", "ge:GetNext", "ge:NPUClearFloatStatusV2", "ge:Identity", "ge:Data",
                      "ge:AssignAdd", "ge:PartitionedCall"]

IGNORE_OPNAME_LIST = ["StreamSend", "StreamRecv"]


Infinity = float("inf")
NaN = float("nan")


class FileNode:
    def __init__(self, name, dirname, level):
        '''
        记录文件/目录信息
        @name: 文件/目录名称
        @dirname: 文件/目录所在绝对路径
        @level: 文件/目录所在层级数
        '''
        self.name = name
        self.abs_path = os.path.join(dirname, name)
        self.isfile = os.path.isfile(self.abs_path)
        self.level = level
        if self.level == 0:
            self.rel_path = f"{name}"
        else:
            self.rel_path = os.path.join(*self.abs_path.split("/")[-level:], name)
        self.subfiles = []


class DirTree:
    def __init__(self, dir_path):
        '''
        记录目录树数信息
        @dir_path: 目录路径
        '''
        self.root_dir = FileNode(os.path.basename(dir_path), os.path.dirname(dir_path), 0)
        self.level_idx = [[], [], [], [], [], [], [], []]
        self.name_idx = {}
        self.init_tree(self.root_dir, 0)

    def init_tree(self, rdir, level):
        '''
        初始化目录, 记录每个文件/目录节点信息, 并按level级别分组
        '''
        self.level_idx[level].append(rdir)
        if rdir.isfile:
            self.name_idx[rdir.name] = rdir
            return
        for d in os.listdir(rdir.abs_path):
            rdir.subfiles.append(FileNode(d, rdir.abs_path, level + 1))
        for subdir in rdir.subfiles:
            self.init_tree(subdir, subdir.level)
        return


class DumpCheck:
    def __init__(self, dump_json_dict, **kwargs):
        '''
        根据dump config分析dump目录格式和数据的正确性, 当配置 iteration="all"时, 需要传入iteration_id_list= \
        '{实际iteration数(acl)/实际step数(ge_option)}', 多卡用例需要传入 expected_device_id_list=[实际device id列表]
        @dump_json_dict: MINDSPORE_DUMP_CONFIG指定的json文件内容字典
        '''
        self.dump_path = dump_json_dict["common_dump_settings"].get("path")
        if not self.dump_path and os.environ.get("MS_DIAGNOSTIC_DATA_PATH"):
            self.dump_path = os.path.join(os.environ["MS_DIAGNOSTIC_DATA_PATH"], "debug_dump")
        self.dump_mode = dump_json_dict["common_dump_settings"]["dump_mode"]
        self.kernels = dump_json_dict["common_dump_settings"]["kernels"]
        self.saved_data = dump_json_dict["common_dump_settings"]["saved_data"]
        self.input_output = dump_json_dict["common_dump_settings"]["input_output"]
        self.net_name = dump_json_dict["common_dump_settings"]["net_name"]
        self.op_debug_mode = dump_json_dict["common_dump_settings"]["op_debug_mode"]
        self.statistic_category = dump_json_dict["common_dump_settings"].get("statistic_category",
                                                                             ["max", "min", "l2norm"])
        if dump_json_dict["common_dump_settings"]["iteration"] == "all":
            iteration_id_list = kwargs.get("iteration_id_list")
            if not iteration_id_list:
                raise Exception("Iteration id list get error, please pass iteration_id_list kwargs.")
            self.iteration_id_list = [str(i) for i in range(0, int(iteration_id_list))]
        else:
            self.iteration_id_list = self.get_iteration_id_list(dump_json_dict["common_dump_settings"]["iteration"])
        # 补充参数
        self.expected_device_id_list = kwargs.get("expected_device_id_list", [str(DEVICE_ID)])
        self.graph_id_list = kwargs.get("graph_id_list", ["0"])
        self.expect_dump_op = []
        if self.dump_mode == 2 or self.op_debug_mode != 0:
            self.expect_dump_op = kwargs.get("expect_dump_op", [])
        self.check_details = kwargs.get("check_details", True)

        self.target_dir = DirTree(self.dump_path)
        self.exceptions = []
        self.net_info = {}
        self.file_map_dict = {}
        self.dump_op_list = []

    @staticmethod
    def get_iteration_id_list(iteration):
        iteration_id_list = []
        for part in iteration.split("|"):
            if "-" in part:
                start_iter, end_iter = part.split("-")[0], part.split("-")[1]
                for i in range(int(start_iter), int(end_iter) + 1):
                    iteration_id_list.append(str(i))
            else:
                iteration_id_list.append(part)
        return iteration_id_list

    @staticmethod
    def check_file_permission_and_size(file_path, expect_mode=FILE_RWGR_MODE):
        # 校验文件权限和是否为空
        if oct(os.stat(file_path).st_mode)[-3:] > expect_mode:
            logger.error(f"file: {file_path} permission check failed")
            return False
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            logger.error(f"file: {file_path} is empty")
            return False
        return True


class SyncDumpCheck(DumpCheck):
    def __init__(self, dump_json_dict, **kwargs):
        super().__init__(dump_json_dict, **kwargs)
        if not dump_json_dict.get("e2e_dump_settings"):
            raise Exception("Dump config dont have e2e_dump_settings but O0 or O1")
        if self.saved_data == "statistic":
            self.stat_calc_mode = dump_json_dict["e2e_dump_settings"].get("stat_calc_mode", "host")
        else:
            self.stat_calc_mode = "host"
        self.sample_mode = dump_json_dict["e2e_dump_settings"].get("sample_mode", 0)
        self.sample_num = dump_json_dict["e2e_dump_settings"].get("sample_num", 100)
        self.save_kernel_args = dump_json_dict["e2e_dump_settings"].get("save_kernel_args", False)
        self.expected_device_id_list = kwargs.get("expected_device_id_list", ['0'])
        self.enable = dump_json_dict["e2e_dump_settings"].get("enable", True)

    def dump_result_check(self):
        # 校验是否有空目录和空文件
        for level_list in self.target_dir.level_idx:
            for level_dir in level_list:
                if not level_dir.isfile and len(level_dir.subfiles) == 0:
                    self.exceptions.append(Exception(f"Dump dir {level_dir.abs_path} is null."))
                    continue
                if level_dir.isfile and not self.check_file_permission_and_size(level_dir.abs_path):
                    self.exceptions.append(Exception(f"Dump file {level_dir.abs_path} is null or high permission."))

        self.check_rank_dir_correct()
        for rank_dir in self.target_dir.level_idx[1]:
            self.check_net_name_correct(rank_dir)
            if self.op_debug_mode == 0:
                self.check_iteration_correct()
        if len(self.exceptions) > 0:
            for expt in self.exceptions:
                print(expt)
            raise self.exceptions[0]

    def check_rank_dir_correct(self):
        # 对应config字段support_device
        expect_rank_dir = {f"rank_{i}" for i in self.expected_device_id_list}
        actual_rank_dir = {rank_dir.name for rank_dir in self.target_dir.level_idx[1]}
        if actual_rank_dir != expect_rank_dir:
            self.exceptions.append(Exception(f"Rank dir lack: {expect_rank_dir - actual_rank_dir}."))

    def check_net_name_correct(self, rank_dir):
        # 对应config字段net_name
        sub_dirs = [sub_dir.name for sub_dir in rank_dir.subfiles]
        if self.net_name not in sub_dirs:
            self.exceptions.append(Exception(f"Net_name wrong, each rank dir contain: {sub_dirs}."))

    def check_iteration_correct(self):
        actual_iteration_list = {d.name for d in self.target_dir.level_idx[4]}
        expect_iteration_list = set(self.iteration_id_list)
        if actual_iteration_list != expect_iteration_list:
            self.exceptions.append(Exception(f"Iteration lack iteration_id:" \
                                                 f" {expect_iteration_list - actual_iteration_list}"))
