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
"""Process bar."""
import sys
import time
from typing import Iterable, Optional, Any


class ProcessBar:
    """
    A progress bar for tracking the progress of an iterable or a process with a known total.
    """

    def __init__(
            self,
            iterable: Optional[Iterable] = None,
            desc: str = "",
            bar_length: int = 20,
            update_interval: float = 0.1,
    ):
        """
        Initialize the ProcessBar.

        Args:
            iterable: An optional iterable to track progress.
            desc: A description of the process being tracked.
            bar_length: The length of the progress bar in characters.
            update_interval: The minimum time interval between progress bar updates.
        """
        if not isinstance(iterable, Iterable):
            raise ValueError("Must provide an iterable")

        if not isinstance(desc, str):
            raise ValueError("desc must be a string")

        if not isinstance(bar_length, int):
            raise ValueError("bar_length must be an integer")

        if bar_length <= 0:
            raise ValueError("bar_length must be greater than 0")

        if not isinstance(update_interval, float):
            raise ValueError("update_interval must be a float")

        if update_interval < 0:
            raise ValueError("update_interval must be greater than 0")

        self.iterable: Iterable = iterable
        self.total: int = len(iterable)
        self.desc: str = desc
        self.bar_length: int = bar_length
        self.update_interval: float = update_interval
        self.current: int = 0
        self.cur_item_name: Optional[str] = None
        self.start_time: float = time.time()
        self.last_update_time: float = self.start_time

    def update(self, n: int = 1, item_name: Optional[str] = None) -> None:
        """
        Update the progress bar.

        Args:
            n: The number of items or steps to increment the progress by.
            item_name: The name of the current item being processed.
        """
        self.current += n
        self.cur_item_name = item_name
        now = time.time()
        if now - self.last_update_time >= self.update_interval:
            self._print_progress(now)
            self.last_update_time = now

    def _print_progress(self, now: float) -> None:
        """
        Print the current progress to the console.

        Args:
            now: The current timestamp.
        """
        elapsed = now - self.start_time
        if self.current > 0 and self.total > 0:
            progress = min(1, self.current / self.total)
            eta = (self.total - self.current) * (elapsed / self.current)
            block = int(round(self.bar_length * progress))

            text = (
                f"\r{self.desc}: [{block * '#' + (self.bar_length - block) * ' '}] "
                f"{self.current}/{self.total} {self.cur_item_name} "
                f"Elapsed: {int(elapsed)}s ETA: {int(eta)}s"
            )
            sys.stdout.write(text)
            sys.stdout.flush()

    def __iter__(self) -> Iterable[Any]:
        """
        Iterate over the items in the iterable, updating the progress bar for each item.

        Yields:
            The next item from the iterable.

        Raises:
            ValueError: If no iterable was provided during initialization.
        """
        if self.iterable is None:
            raise ValueError("Must provide an iterable")
        for item in self.iterable:
            yield item
            self.update(item_name=item.__class__.__name__)
        sys.stdout.write("\n")
