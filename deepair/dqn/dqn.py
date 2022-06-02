
from deepair.core.base_algo import BaseAlgo


class DQN(BaseAlgo):
    def __init__(self, config):
        super().__init__(config=config)

    def config_parser(self, config):
        pass
