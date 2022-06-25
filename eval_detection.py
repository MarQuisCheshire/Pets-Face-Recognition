import warnings
from pathlib import Path

import torch

from engine import DetectionController
from utils import configure_trainer, get_config

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    lightning_logger = False
    checkpoint_path = Path('results')

    config = get_config(Path('configs/mask/mask_rcnn_config.py'))

    controller = DetectionController(config=config)
    controller.load_state_dict(torch.load(Path('configs/to_reproduce/mask/epoch=64.ckpt')))

    trainer = configure_trainer(config, lightning_logger, checkpoint_path)

    trainer.test(controller)

    print('Completed!')
