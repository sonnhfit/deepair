# pylint: disable=W,C,R
import numpy as np


class BaseCallback:
    def __init__(self) -> None:
        self.n_calls = 0
        self.num_timesteps = 0
        self.model = None

    def init_callback(self, model):
        self.model = model


    def on_step(self, model) -> bool:
        self.n_calls += 1

        return True


class EvalCallback(BaseCallback):
    def __init__(self, env, best_model_save_path, eval_freq) -> None:
        super().__init__()

        self.env = env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.best_score = -np.inf

    
    def on_step(self, model) -> bool:
        self.n_calls += 1

        if self.n_calls % self.eval_freq == 0:
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = model.select_action(state, deterministic=True)
                #print(action)
                next_state, reward, done, _ = self.env.step(int(action))

                state = next_state
                score += reward

            if score > self.best_score:
                # save model
                model.save(self.best_model_save_path)
                self.best_score = score
                print("Model saved. Score: ", score)

        return True
        
