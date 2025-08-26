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
# ==============================================================================
"""Distributed sampler module."""

import math
from collections.abc import Iterator
from typing import Optional, TypeVar

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.dataset.dataloader.dataset import Dataset
from mindspore.dataset.dataloader.sampler import Sampler


__all__ = ["DistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    Example::
        >>> sampler = DistributedSampler(dataset)
        >>> loader = DataLoader(dataset, shuffle=None, sampler=sampler
        ...                     num_replicas=2, rank=0)
        >>> for data in loader:
        ...     print(data)
    """

    def __init__(
            self,
            dataset: Dataset,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: Optional[bool] = True,
            seed: Optional[int] = 0,
            drop_last: Optional[bool] = False,
    ) -> None:
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("MindSpore distributed feature is not available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("MindSpore distributed feature is not available.")
            rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError(
                f"Invalid num_replicas: {num_replicas}, num_replicas should be greater than 0."
            )
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank: {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # If the dataset length is evenly divisible by replicas or not to drop
        if len(self.dataset) % self.num_replicas == 0 or not self.drop_last:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        self.total_samples = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = ms.Generator()
            g.manual_seed(self.seed + self.epoch)
            # TODO: need to use mint operator on cpu backend
            indices = ms.ops.randperm(len(self.dataset), seed=self.seed + self.epoch).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data
            indices = indices[: self.total_samples]
        assert len(indices) == self.total_samples

        # subsample
        indices = indices[self.rank : self.total_samples : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
