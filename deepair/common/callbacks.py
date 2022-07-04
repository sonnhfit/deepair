# pylint: disable=W,C,R

class BaseCallback:
    def __init__(self) -> None:
        self.n_calls = 0
        self.num_timesteps = 0
        self.model = None

    def init_callback(self, model):
        self.model = model


    def on_step(self) -> bool:

        self.n_calls += 1

        return True
