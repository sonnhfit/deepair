.. deepair documentation master file, created by
   sphinx-quickstart on Tue May 31 02:38:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Deepair's documentation!
===================================

**Deepair** (/diːp er/) is a Python library that have reliable implementations of Deep reinforcement learning algorithms.


```python
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
```
![rainbow lunalander env](docs/source/_static/img/rainbow_lunalander.gif)
.. toctree::
   :maxdepth: 1

   install
   tutorial
   rlalgorithms
   custom_technique
   developer_guides
   modules





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
