# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Profiler path manager"""
import os
import re
import shutil
import warnings

from mindspore import log as logger
from mindspore.profiler.parser.ascend_analysis.constant import Constant


__all__ = ['PathManager']


class PathManager:
    """
    Path common operations manager
    """
    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750

    @classmethod
    def check_input_directory_path(cls, path: str):
        """
        Function Description:
            check whether the path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isfile(path):
            msg = "Invalid input path is a file path: {path}"
            raise RuntimeError(msg)

    @classmethod
    def check_input_file_path(cls, path: str):
        """
        Function Description:
            check whether the file path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the file path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isdir(path):
            msg = "Invalid input path is a directory path: {path}"
            raise RuntimeError(msg)

    @classmethod
    def copy_file(cls, src_path: str, dst_path: str):
        """
        Function Description:
            copy file safety
        Parameter:
            src_path: file source path
            dst_path: file destination path
        Exception Description:
            when src_path is link throw exception
        """
        if not os.path.exists(src_path):
            logger.warning("The source file does not exist: %s", src_path)
            return

        cls.check_input_file_path(src_path)
        dst_dir = os.path.dirname(dst_path)
        cls.check_directory_path_writeable(dst_dir)

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as err:
            msg = f"Failed to copy path: {src_path}"
            raise RuntimeError(msg) from err

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_writeable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.W_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        """
        Function Description:
            remove path safety
        Parameter:
            path: the path to remove
        Exception Description:
            when invalid data throw exception
        """
        msg = f"Failed to remove path: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def remove_file_safety(cls, file: str):
        """
        Function Description:
            remove file safety
        Parameter:
            path: the file to remove
        Exception Description:
            when invalid data throw exception
        """
        msg = f"Failed to remove file: {file}"
        if os.path.islink(file):
            raise RuntimeError(msg)
        if not os.path.exists(file):
            return
        try:
            os.remove(file)
        except FileExistsError:
            return
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def make_dir_safety(cls, path: str):
        """
        Function Description:
            make directory safety
        Parameter:
            path: the directory to remove
        Exception Description:
            when invalid data throw exception
        """
        msg = f"Failed to make directory: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.makedirs(path, mode=cls.DATA_DIR_AUTHORITY, exist_ok=True)
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def create_file_safety(cls, path: str):
        """
        Function Description:
            create file safety
        Parameter:
            path: the file to remove
        Exception Description:
            when invalid data throw exception
        """
        msg = f"Failed to create file: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def _input_path_common_check(cls, path: str):
        """
        Function Description:
            input path check common function
        Parameter:
            path: the file path to check
        Exception Description:
            when invalid data throw exception
        """
        if len(path) > cls.MAX_PATH_LENGTH:
            raise RuntimeError("Length of input path exceeds the limit.")

        if os.path.islink(path):
            msg = f"Invalid input path is a soft chain: {path}"
            raise RuntimeError(msg)

        pattern = r'(\.|/|_|-|\s|[~0-9a-zA-Z])+'
        if not re.fullmatch(pattern, path):
            msg = f"Invalid input path: {path}"
            raise RuntimeError(msg)

        path_split_list = path.split("/")
        for name in path_split_list:
            if len(name) > cls.MAX_FILE_NAME_LENGTH:
                raise RuntimeError("Length of input path exceeds the limit.")

    @classmethod
    def get_profiler_parent_path_list(cls, input_path: str):
        """
        Function Description:
            get valid profiler parent path list from input_path
        Parameter:
            input_path: The directory path from which to extract profiler parent paths.
        Return:
            A list containing the input path or its subdirectories that are valid profiler parents.
        """
        profiler_path = os.path.join(input_path, Constant.PROFILER_DIR)
        if os.path.isdir(profiler_path) and (cls.get_fwk_path(profiler_path) or cls.get_cann_path(profiler_path)):
            return [input_path]
        sub_dirs = os.listdir(os.path.realpath(input_path))
        profiler_parent_path_list = []
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir, Constant.PROFILER_DIR)
            if not os.path.isdir(sub_path):
                continue
            if cls.get_fwk_path(sub_path) or cls.get_cann_path(sub_path):
                profiler_parent_path_list.append(os.path.join(input_path, sub_dir))
        return profiler_parent_path_list

    @classmethod
    def get_fwk_path(cls, input_path: str):
        """
        Function Description:
            get valid framework path from input_path
        Parameter:
            input_path: the directory path to check whether exist valid FRAMEWORK path
        Return:
            The path to the FRAMEWORK directory if found, otherwise an empty string.
        """
        fwk_path = os.path.join(input_path, Constant.FRAMEWORK_DIR)
        if os.path.isdir(fwk_path):
            return fwk_path
        return ""

    @classmethod
    def get_cann_path(cls, input_path: str):
        """
        Function Description:
            get valid PROF_XXX path from input_path
        Parameter:
            input_path: the directory path to check valid PROF_XXX path
        Return:
            The path to the PROF_XXX directory if it matches the pattern and exists, otherwise an empty string.
        """
        sub_dirs = os.listdir(os.path.realpath(input_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^PROF_\d+_\d+_[0-9a-zA-Z]+", sub_dir):
                return sub_path
        return ""
