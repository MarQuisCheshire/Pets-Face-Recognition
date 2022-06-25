import warnings
from pathlib import Path

import torch

from engine import Controller
from utils import configure_trainer, get_config

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    lightning_logger = False
    checkpoint_path = Path('results')

    config = get_config(Path('configs/to_reproduce/cat_fe/cat_fe_head.py'))

    controller = Controller(config=config)
    controller.load_state_dict(
        torch.load(Path('configs/to_reproduce/cat_fe/epoch=42_head.ckpt')),
        strict=False
    )

    trainer = configure_trainer(config, lightning_logger, checkpoint_path)

    trainer.test(controller)

    print('Completed!')
