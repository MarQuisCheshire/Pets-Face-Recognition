import inspect
import os
from contextlib import suppress
from importlib.util import module_from_spec, spec_from_file_location
from typing import List, Union, Optional

import torch
from pytorch_lightning.plugins import TrainingTypePlugin, DDPPlugin

from engine import Trainer


class DictWrapper:
    def __init__(self, d=None):
        if d:
            for k in d:
                if not inspect.ismodule(d[k]):
                    self.__setattr__(k, d[k])

    def __getitem__(self, index):
        return self.__getattribute__(index)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return 'DictWrapper: ' + repr(self.__dict__)

    def __getattr__(self, item):
        return getattr(self.__dict__, item)


class _SingletonBase(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(DictWrapper, metaclass=_SingletonBase):
    def __repr__(self):
        return 'Config: ' + repr(self.__dict__)


def get_dict_wrapper(path) -> DictWrapper:
    assert os.path.exists(path)
    spec = spec_from_file_location('config', path)
    config = module_from_spec(spec)
    spec.loader.exec_module(config)
    config = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    config = DictWrapper(config)
    return config


def get_config(path) -> Config:
    assert os.path.exists(path)
    spec = spec_from_file_location('config', path)
    config = module_from_spec(spec)
    spec.loader.exec_module(config)
    config = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    if Config in _SingletonBase._instances:
        del _SingletonBase._instances[Config]
    config = Config(config)
    return config


def get_gpus(world_size=1) -> Union[int, List[int]]:
    assert world_size <= torch.cuda.device_count(), f"Only {torch.cuda.device_count()} are visible"
    assert world_size >= 0
    if world_size == 0:
        return 0
    gpus = []
    for i in range(torch.cuda.device_count()):
        with suppress(RuntimeError):
            torch.zeros(5, 5, device=f'cuda:{i}')
            gpus.append(i)

        if len(gpus) >= world_size:
            return gpus
    raise Exception(f'Cannot access {world_size} gpus')


def parse_gpus(cfg: Config) -> Union[int, List[int]]:
    if cfg.get('distributed_train'):
        if isinstance(cfg.device, list):
            gpus = cfg.device
            assert cfg.world_size == len(gpus), 'Not enough GPUs'
        else:
            gpus = cfg.world_size
    elif cfg.device == 'cpu':
        gpus = 0
    elif cfg.device == 'cuda':
        gpus = get_gpus()
    else:
        gpus = [int(cfg.device.split(':')[-1])]
        gpus = list({i for i in gpus if i < torch.cuda.device_count()})
        if len(gpus) == 0:
            gpus = 0 if not torch.cuda.is_available() else [0]
    return gpus


def is_main_process() -> bool:
    return all(os.environ.get(i, 0) == 0 for i in ('NODE_RANK', 'LOCAL_RANK'))


def get_strategy(config: Config) -> Optional[TrainingTypePlugin]:
    if config.get('distributed_train', False):
        return DDPPlugin(
            find_unused_parameters=config.get('find_unused_parameters', False),
            gradient_as_bucket_view=config.get('gradient_as_bucket_view', True)
        )


def configure_trainer(config: Config, lightning_logger, lightning_log_dir=None) -> Trainer:
    trainer = Trainer(
        gpus=parse_gpus(config),
        default_root_dir=lightning_log_dir,
        strategy=get_strategy(config),
        max_epochs=config.n_epochs,
        logger=lightning_logger if lightning_logger is not None else False,
        enable_checkpointing=True,
        callbacks=config.get('callbacks'),
        **config.get('trainer_kwargs', {})
    )

    return trainer


def find_max_batch_size(trainer: Trainer, model) -> Optional[int]:
    new_batch_size = trainer.tuner.scale_batch_size(model, **model.config.cfg.get('find_max_batch_size_kwargs', {}))
    if new_batch_size is not None:
        print(f'Computed new batch size = {new_batch_size}')
        return new_batch_size


def find_optimal_init_lr(trainer: Trainer, model) -> float:
    assert len(model.config.cfg.opt_params) == 1, 'find_optimal_init_lr supports only one optimizer'
    new_lr = trainer.tuner.lr_find(model, **model.config.cfg.get('find_optimal_init_lr_kwargs', {})).suggestion()
    print(f'Computed new init lr = {new_lr}')
    return new_lr
