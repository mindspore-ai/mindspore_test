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

"""compile custom kernel with ninja"""

import os
import shlex
import subprocess
import sysconfig
import time
import stat
from mindspore import log as logger


class VersionManager:
    """version manager"""

    def __init__(self):
        self.entries = {}  # module_name : (version, hash)

    def _get_version(self, module_name):
        """get version"""
        return self.entries.get(module_name, (None, None))[0]

    def _update_version_if_changed(self, module_name, sources, build_args, build_dir):
        """update version if changed"""
        hash_value = self._update_hash(0, build_dir)
        hash_value = self._update_sources_hash(hash_value, sources)
        hash_value = self._update_args_hash(hash_value, build_args)

        entry = self.entries.get(module_name)
        if entry is None:
            self.entries[module_name] = entry = (0, hash_value)
        elif hash_value != entry[1]:
            self.entries[module_name] = entry = (entry[0] + 1, hash_value)

        return entry[0]

    def _update_hash(self, seed, value):
        """update hash value"""
        # Good old boost::hash_combine
        return seed ^ (hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2))

    def _update_sources_hash(self, hash_value, sources):
        """hash source files"""
        for filename in sources:
            with open(filename) as file:
                hash_value = self._update_hash(hash_value, file.read())
        return hash_value

    def _update_args_hash(self, hash_value, build_args):
        """hash build arguments"""
        for group in build_args:
            if group:
                for argument in group:
                    hash_value = self._update_hash(hash_value, argument)
        return hash_value

    def check_version(self, name, sources, cflags, ldflags, include_paths, build_dir):
        """check version"""
        old_version = self._get_version(name)
        version = self._update_version_if_changed(name, sources, [cflags, ldflags, include_paths], build_dir)
        logger.info(f'Build module {name}, version={version}')
        if version > 0:
            if version != old_version:
                logger.info(
                    f'The conditions for extension module {name} have changed. '
                    f'Updating to version {version} and re-building as {name}_v{version}.'
                )
            name = f'{name}_v{version}'

        if version != old_version:
            return True
        logger.info(f'No modifications detected for extension module {name}')
        return False


version_manager = VersionManager()


class FileLocker:
    """FileLocker"""

    def __init__(self, build_dir):
        """FileLocker"""
        self.lock_file_name = os.path.join(build_dir, 'build.lock')
        self.lock_fd = None

    def try_lock(self):
        """Acquire a file-based lock."""
        try:
            mode = stat.S_IRUSR | stat.S_IWUSR
            self.lock_fd = os.open(self.lock_file_name, os.O_CREAT | os.O_EXCL, mode)
            return True
        except FileExistsError:
            return False

    def release_lock(self):
        """Release the file-based lock."""
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        os.remove(self.lock_file_name)

    def wait(self):
        """Wait until lock is released."""
        while os.path.exists(self.lock_file_name):
            time.sleep(0.5)


class ExtensionBuilder:
    """ExtensionBuilder"""

    def __init__(self, build_dir):
        """ExtensionBuilder"""
        self.build_dir = build_dir

    def _compile(self, name, sources, cflags, ldflags, include_paths):
        """Compile."""
        if version_manager.check_version(name, sources, cflags, ldflags, include_paths, self.build_dir):
            locker = FileLocker(self.build_dir)
            if locker.try_lock():
                try:
                    self._write_ninja_file_and_build_library(name, sources, cflags, ldflags, include_paths)
                finally:
                    locker.release_lock()
            else:
                locker.wait()
        logger.info(f'Loading extension module {name}...')

    def _verify_ninja_availability(self):
        """Check ninja is available."""
        try:
            subprocess.check_output('ninja --version'.split())
        except Exception:
            raise RuntimeError("Ninja is required to load C++ extensions")

    def _write_ninja_file_and_build_library(self, module_name, sources, cflags, ldflags, include_paths):
        """Write ninja file and build library."""
        self._verify_ninja_availability()

        ninja_build_file = os.path.join(self.build_dir, 'build.ninja')
        logger.info(f'Save ninja build file {ninja_build_file}.')
        self._write_ninja_file(ninja_build_file, module_name, sources, cflags, ldflags, include_paths)

        logger.info(f'Building extension module {module_name}.')
        self._run_ninja_build(module_name)

    def _write_ninja_file(self, fname, name, sources, extra_cflags, extra_ldflags, extra_include_paths):
        """Write ninja file."""
        python_include_path = sysconfig.get_path('include', scheme='posix_prefix')
        python_includes = [python_include_path] if python_include_path is not None else []
        cflags = [f'-DMS_EXTENSION_NAME={name}', "-D_GLIBCXX_USE_CXX11_ABI=0"]
        cflags += [f'-I{shlex.quote(os.path.abspath(include.strip()))}' for include in extra_include_paths]
        cflags += [f'-isystem {shlex.quote(include)}' for include in python_includes]
        cflags += ['-fPIC', '-std=c++17']
        cflags += extra_cflags
        cflags = [flag.strip() for flag in cflags]

        # '/path/to/file.cpp' -> 'file'
        objs = [os.path.splitext(os.path.basename(src))[0] + ".o" for src in sources]
        sources = [os.path.abspath(file) for file in sources]
        ldflags = ['-shared'] + [flag.strip() for flag in extra_ldflags]
        target = name + '.so'

        config = ['ninja_required_version = 1.3']
        config.append('cxx = ' + os.environ.get('CXX', 'g++'))

        flags = [f'cflags = {" ".join(cflags)}']
        flags.append(f'ldflags = {" ".join(ldflags)}')

        compile_rule = ['rule compile']
        compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

        build = [f'build {obj.replace(" ", "$ ")}: compile {src.replace(" ", "$ ")}' for src, obj in zip(sources, objs)]

        link_rule = ['rule link', '  command = $cxx $in $ldflags -o $out']
        link = [f'build {target}: link {" ".join(objs)}']
        default = [f'default {target}']

        blocks = [config, flags, compile_rule, link_rule, build, link, default]
        content = "\n\n".join("\n".join(b) for b in blocks) + "\n"

        if os.path.exists(fname):
            with open(fname) as f:
                old_content = f.read()
            if old_content == content:
                return

        with open(fname, 'w') as source_file:
            source_file.write(content)

    def _run_ninja_build(self, module_name):
        """Run ninja build."""
        cmd = ['ninja', '-v']
        env = os.environ.copy()

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.build_dir, check=True, env=env)
            # If the build succeeds, do nothing with the output (silent)
        except subprocess.CalledProcessError as e:
            # Capture the error details
            stderr_output = e.stderr.decode() if e.stderr else ""
            stdout_output = e.stdout.decode() if e.stdout else ""
            full_output = stderr_output + stdout_output

            # Format the error message
            msg = f"Error building extension '{module_name}': {full_output}"

            # In multi-card situation, only one process build the library.
            # When building failed, the old extension library should be removed.
            so_file = os.path.join(self.build_dir, f"{module_name}.so")
            if os.path.exists(so_file):
                os.remove(so_file)
            raise RuntimeError(msg) from e

    def build(self, module_name, sources, extra_cflags=None, extra_ldflags=None, extra_include_paths=None):
        """Build module."""
        src = [sources] if isinstance(sources, str) else sources
        self._compile(module_name, src, extra_cflags, extra_ldflags, extra_include_paths)
        return os.path.join(self.build_dir, f"{module_name}.so")
