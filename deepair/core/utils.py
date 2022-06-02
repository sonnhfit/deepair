# pylint: disable=W, C, E, R

from typing import Union
import gym
import numpy as np


def is_vectorized_box_observation(
    observation: np.ndarray, observation_space: gym.spaces.Box
) -> bool:
    """
    For box observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + f"Box environment, please use {observation_space.shape} "
            + "or (n_env, {}) for the observation shape.".format(", ".join(
                map(str, observation_space.shape)))
        )


def is_vectorized_discrete_observation(
    observation: Union[int, np.ndarray],
    observation_space: gym.spaces.Discrete
) -> bool:
    """
    For discrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if isinstance(observation, int) or observation.shape == ():
        # A numpy array of a number, has shape empty tuple '()'
        return False
    elif len(observation.shape) == 1:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + "Discrete environment, please use () or (n_env,) for the observation shape."
        )


def is_vectorized_multidiscrete_observation(
    observation: np.ndarray,
    observation_space: gym.spaces.MultiDiscrete
) -> bool:
    """
    For multidiscrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == (len(observation_space.nvec),):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
            + f"environment, please use ({len(observation_space.nvec)},) or "
            + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
        )


def is_vectorized_multibinary_observation(
    observation: np.ndarray,
    observation_space: gym.spaces.MultiBinary
) -> bool:
    """
    For multibinary observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == (observation_space.n,):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
            + f"environment, please use ({observation_space.n},) or "
            + f"(n_env, {observation_space.n}) for the observation shape."
        )


def is_vectorized_dict_observation(
    observation: np.ndarray, observation_space: gym.spaces.Dict) -> bool:
    """
    For dict observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    # We first assume that all observations are not vectorized
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        # This fails when the observation is not vectorized
        # or when it has the wrong shape
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    # Now we check that all observation are vectorized and have the correct shape
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        # Retrieve error message
        error_msg = ""
        try:
            is_vectorized_observation(observation[key], observation_space.spaces[key])
        except ValueError as e:
            error_msg = f"{e}"
        raise ValueError(
            f"There seems to be a mix of vectorized and non-vectorized observations. "
            f"Unexpected observation shape {observation[key].shape} for key {key} "
            f"of type {observation_space.spaces[key]}. {error_msg}"
        )



def is_vectorized_observation(
    observation: Union[int, np.ndarray], observation_space: gym.spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.
    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """

    is_vec_obs_func_dict = {
        gym.spaces.Box: is_vectorized_box_observation,
        gym.spaces.Discrete: is_vectorized_discrete_observation,
        gym.spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
        gym.spaces.MultiBinary: is_vectorized_multibinary_observation,
        gym.spaces.Dict: is_vectorized_dict_observation,
    }

    for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
        if isinstance(observation_space, space_type):
            return is_vec_obs_func(observation, observation_space)
    else:
        # for-else happens if no break is called
        raise ValueError(
            f"Error: Cannot determine if the observation is" 
            + f"vectorized with the space type {observation_space}."
        )
