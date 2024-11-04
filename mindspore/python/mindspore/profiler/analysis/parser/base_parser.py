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
"""Base parser"""
from typing import (
    Any,
    List,
    Callable,
    Optional,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from mindspore import log as logger


class BaseParser:
    """
    Base class for all parsers in the workflow.
    """

    def __init__(self, next_parser: Optional["BaseParser"] = None):
        """
        Initialize the BaseParser.

        Args:
            next_parser (Optional[BaseParser]): The next parser in the chain.
        """
        self.next_parser: Optional["BaseParser"] = next_parser
        self._post_hooks: List[Callable[[Any], None]] = []

    def set_next(self, next_parser: "BaseParser") -> "BaseParser":
        """
        Set the next parser in the chain.

        Args:
            next_parser (BaseParser): The next parser to be set.

        Returns:
            BaseParser: The next parser that was set.
        """
        if not isinstance(next_parser, BaseParser):
            raise ValueError(
                f"next_parser {next_parser.__class__.__name__} must be a BaseParser"
            )
        self.next_parser = next_parser
        return self

    def parse(self, data: Any) -> Any:
        """
        Parse the input data and execute post-hooks.

        Args:
            data (Any): The input data to be parsed.

        Returns:
            Any: The parsed result.
        """
        result = self._parse(data)
        self._execute_post_hooks(result)
        return result

    def register_post_hook(self, hook: Callable[[Any], None]) -> "BaseParser":
        """
        Register a post-hook to be executed after parsing.

        Args:
            hook (Callable[[Any], None]): The hook function to be registered.

        Returns:
            BaseParser: The current parser instance.

        Raises:
            ValueError: If the hook is not callable.
        """
        if callable(hook):
            self._post_hooks.append(hook)
        else:
            raise ValueError("Hook must be callable")
        return self

    def _execute_post_hooks(self, res: Any) -> None:
        """
        Execute all registered post-hooks asynchronously.

        Args:
            res (Any): The result to be passed to the post-hooks.
        """
        if not self._post_hooks:
            return

        with ProcessPoolExecutor() as executor:
            future_to_hook = {
                executor.submit(hook, res): hook.__name__
                for hook in self._post_hooks
            }

            for future in as_completed(future_to_hook):
                hook_name = future_to_hook[future]
                try:
                    future.result()
                    logger.info("Hook %s executed successfully", hook_name)
                except Exception as exc: # pylint: disable=W0703
                    logger.error("Hook %s executed failed: %s", hook_name, exc)

    def _parse(self, data: Any) -> Any:
        """
        Abstract method to be implemented by subclasses for parsing logic.

        Args:
            data (Any): The input data to be parsed.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")
