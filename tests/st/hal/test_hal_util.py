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
import subprocess
from subprocess import TimeoutExpired, CalledProcessError


def run_cmd(cmd, cwd=None, timeout=None, raise_exec=True):
    """
    Run Linux command and return result

    Args:
        cmd (str|list): Command to execute, can be string or argument list
        cwd (str|None): Working directory, default is None which means using current directory
        timeout (int|None): Timeout (seconds), default is None which means no timeout limit
        raise_exec (bool): Whether to raise exception when command execution fails, default is True

    Returns:
        tuple: (return_code, stdout, stderr)
            - return_code (int): Command return code, 0 means success
            - stdout (str): Standard output
            - stderr (str): Standard error

    Example:
        >>> return_code, stdout, stderr = run_cmd("ls -l")
        >>> print(f"Return code: {return_code}")
        >>> print(f"Output: {stdout}")
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except TimeoutExpired as e:
        err_msg = f"Command timed out: {str(e)}"
        print(err_msg)
        if raise_exec:
            raise e
        return -1, "", err_msg
    except CalledProcessError as e:
        err_msg = f"Command failed with return code {e.returncode}: {str(e)}"
        print(err_msg)
        if raise_exec:
            raise e
        return -1, "", err_msg
