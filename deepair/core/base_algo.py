from abc import ABC, abstractmethod


class BaseAlgo(ABC):

    def __init__(self, config):

        self.policy = None
        self.env = None
        self._logger = None
        self._config_parser(config)

    def _config_parser(self, config):
        pass

    @abstractmethod
    def train(self):
        pass

    def predict(self):
        pass

    def get_env(self):
        pass

    def set_env(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
