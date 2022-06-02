
from deepair.core.base_algo import BaseAlgo


class DQN(BaseAlgo):
    def __init__(self, config):
        super().__init__(config=config)

    def config_parser(self, config) -> None:
        """
        {
            "num_workers": 0,
            "num_gpus": 0,
            "hiddens": [128, 128],
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "train_batch_size": 32,
            "lr": 0.00001,
            "replay_buffer_size": 10000,
            "gamma": 0.99,
            "exploration_initial_eps": 1,
            "exploration_final_eps": 0.1
            "target_network_update_freq": 1,
            "use_huber": True,
            "huber_threshold": 0.1,
            "grad_clip": None,
        }
        """

        self.prioritized_replay = config.get('prioritized_replay')
        self.hiddens = config.get('hiddens')
        self.lr = config.get('lr') # pylint: disable=C0103
