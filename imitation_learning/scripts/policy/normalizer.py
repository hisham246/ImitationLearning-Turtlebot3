from typing import Union, Dict
import zarr
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.append('/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/scripts')
from policy.pytorch_util import dict_apply
from policy.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    def __init__(self):
        super().__init__()
        self.params_dict = nn.ParameterDict()  # Ensure params_dict is initialized

    @torch.no_grad()
    def fit(self,
            data: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        """
        Initializes normalization parameters for each key in `data`.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                # Initialize params for each field, like 'obs', 'action', etc.
                self.params_dict[key] = _fit(
                    value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset
                )
        else:
            # If data is not a dictionary, use '_default' as the key
            self.params_dict['_default'] = _fit(
                data, 
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset
            )

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if key not in self.params_dict:
                    print(f"Warning: Key '{key}' not initialized in normalizer's parameters.")
                    continue
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Normalizer is not initialized for non-dict data with '_default' parameters.")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Normalizer has not been initialized.")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = {}
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)


def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    # Create parameter dict
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    x = x * scale + offset if forward else (x - offset) / scale
    return x.reshape(src_shape)