# pylint: disable=W,C,R
import pathlib
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict
import gym
from deepair.common.utils import recursive_getattr, recursive_setattr, save_to_zip_file

class BaseModel(ABC):

    @abstractmethod
    def train(self, timesteps: int, plotting_interval: int=200, eval_env: gym.Env=None):
        """
        training process
        """
    
    @abstractmethod
    def load(self, path: Union[str, pathlib.Path], env: gym.Env=None, device: str='cpu'):
        pass
    
    @abstractmethod
    def save(self, path: Union[str, pathlib.Path]):
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)
        
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(
            path, data=data, params=params_to_save,
            pytorch_variables=pytorch_variables
        )


    def _excluded_save_params(self) -> List[str]:
        return [
            "device",
            "env",
            "eval_env",
            "memory",
            "memory_n",
            "is_test"
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.
        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.
        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["net"]

        return state_dicts, []


    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).
        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

