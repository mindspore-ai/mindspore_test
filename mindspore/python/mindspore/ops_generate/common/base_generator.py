"""
Base generator interface for other pyboost file generator classes
"""

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, **kwargs):
        raise NotImplementedError
