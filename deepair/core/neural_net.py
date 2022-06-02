# pylint: disable=W, C, E, R
from typing import List, Type, Union, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from torch import nn # pylint: disable=import-error
import torch as th
import gym
from gym import spaces
import numpy as np
import copy
from torch.nn import functional as F

from deepair.core.save_utils import get_device
from deepair.core.utils import is_vectorized_observation


TensorDict = Dict[Union[str, int], th.Tensor]


def make_neural_net(
    input_shape: int,
    output_shape: int,
    hidden: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU
) -> List[nn.Module]:

    if len(hidden) > 0:
        modules = [nn.Linear(input_shape, hidden[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(hidden) - 1):
        modules.append(nn.Linear(hidden[idx], hidden[idx + 1]))
        modules.append(activation_fn())

    if output_shape > 0:
        last_layer = hidden[-1] if len(hidden) > 0 else input_shape
        modules.append(nn.Linear(last_layer, output_shape))

    return modules



def obs_as_tensor(
    obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.
    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs).to(device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


class BaseNN(nn.Module, ABC):

    def __init__(self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.
        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            # normalize_images=self.normalize_images,
        )


    def save(self, path: str) -> None:
        """
        Save model to a given location.
        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)


    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> "BaseNN":
        """
        Load model from path.
        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model


    def obs_to_tensor(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)
        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def preprocess_obs(
        self,
        obs: th.Tensor,
        observation_space: gym.spaces.Space
    ) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        """
        Preprocess observation to be to a neural network.
        For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
        For discrete observations, it create a one hot vector.
        :param obs: Observation
        :param observation_space:
        :return:
        """

        if isinstance(observation_space, spaces.Box):
            return obs.float()

        elif isinstance(observation_space, spaces.Discrete):
            # One hot encoding and convert to float to avoid errors
            return F.one_hot(obs.long(), num_classes=observation_space.n).float()

        elif isinstance(observation_space, spaces.MultiDiscrete):
            # Tensor concatenation of one hot encodings of each Categorical sub-space
            return th.cat(
                [
                    F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                    for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
                ],
                dim=-1,
            ).view(obs.shape[0], sum(observation_space.nvec))

        elif isinstance(observation_space, spaces.MultiBinary):
            return obs.float()

        elif isinstance(observation_space, spaces.Dict):
            # Do not modify by reference the original observation
            preprocessed_obs = {}
            for key, _obs in obs.items():
                preprocessed_obs[key] = self.preprocess_obs(_obs, observation_space[key])
            return preprocessed_obs

        else:
            raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
