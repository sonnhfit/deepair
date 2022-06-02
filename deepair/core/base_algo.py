import io
import pathlib
from abc import ABC, abstractmethod
from typing import Union


class BaseAlgo(ABC):

    def __init__(self, config):

        self.policy_network = None
        self.env = None
        self._logger = None
        self.gpu_number = 0
        self.device = 'cpu'
        self.config_parser(config)

    @abstractmethod
    def config_parser(self, config) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    def learn(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def get_env(self):
        return self.env

    def set_env(self, env, reset: bool = True) -> None:
        self.env = env
        if reset:
            self.env.reset()

    def load(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase]
    ):
        pass

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase]
    ):
        pass
