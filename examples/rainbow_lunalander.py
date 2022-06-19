# pylint: disable=C

import gym
from deepair.dqn import Rainbow

env = gym.make('LunarLander-v2')

rain = Rainbow(env=env, memory_size=10000, batch_size=32, target_update=256)

rain.train(timesteps=200000)

# test
state = env.reset()
done = False
score = 0

while not done:
    action = rain.select_action(state, deterministic=True)
    next_state, reward, done, info = env.step(action)

    state = next_state
    score += reward

print("score: ", score)
