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
"""Ms jit stream class"""


class MsJitStream():
    r"""
    Wrapper around a device stream.
    """

    def __init__(self):
        self.init_finished = True

    def record_event(self, event=None):
        event.record(self)
        return event

    def wait_event(self, event):
        event.wait(self)

    def wait_stream(self, stream):
        self.wait_event(stream.record_event())


class CtxAddAttr:
    r"""
    Provide a class for setting attributes.
    """
    def __init__(self):
        self.attrs = {}

    def add_attr(self, name, value):
        self.attrs[name] = value


class MsJitStreamCtx(CtxAddAttr):
    r"""
    Context-manager that selects a given stream.

    Args:
        ctx_stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """

    def __init__(self, ctx_stream):
        super().__init__()
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
