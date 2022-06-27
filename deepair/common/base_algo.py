# pylint: disable=W,C,R
import pathlib
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Iterable, Optional, Any
import gym
from deepair.common.utils import (
    recursive_getattr, recursive_setattr,
    save_to_zip_file, load_from_zip_file, 
    check_for_correct_spaces
)

import torch as th


class BaseAlgo(ABC):
    def __init__(
        self, 
        env: gym.Env = None
    ):
        self.env = env
        self.obs_dim = 1
        self.action_dim = 1

    
    @classmethod
    def load(cls,
        path: Union[str, pathlib.Path], 
        env: gym.Env=None, 
        device: str='cpu',
        custom_objects: Optional[Dict[str, Any]] = None):
        
        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects
        )

        if env is not None:
            # Wrap first if needed
            env = env
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pylint: disable=undefined-variable
            env=env
        )
        if data != None:
            if "obs_dim" in data and "action_dim" in data:
                model.obs_dim = data['obs_dim']
                model.action_dim = data['action_dim']

            # load parameters
            model.__dict__.update(data)

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        return model

    
    
    def save(self, path: Union[str, pathlib.Path],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None
        ):
        data = self.__dict__.copy()

        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self.get_torch_save_params()
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
            "eval_env",
            "env",
            "memory",
            "memory_n",
            "is_test"
        ]

    def get_torch_save_params(self) -> Tuple[List[str], List[str]]:
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
        state_dicts = []

        return state_dicts, []


    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).
        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self.get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).
        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self.get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
