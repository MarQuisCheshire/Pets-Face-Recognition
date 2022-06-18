import argparse
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
from pytorch_lightning.loggers import MLFlowLogger

from engine import DetectionController
from utils import is_main_process, configure_trainer, find_max_batch_size, find_optimal_init_lr, get_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        required=True,
        type=Path,
        help='Path to config file'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    args = parse_args()
    config = get_config(args.config)

    checkpoint_path = None
    lightning_logger = None
    lightning_log_dir = None
    if is_main_process():

        restime = datetime.now().strftime("%Y%m%d-%H%M%S")

        run_output_root = config.output / restime
        config.output = run_output_root
        checkpoint_path = run_output_root / 'checkpoints'
        lightning_log_dir = str((run_output_root / 'lightning').resolve())
        experiment_name = config.get('experiment_name', 'default')
        config.checkpoint_path = checkpoint_path
        config.img_dir = run_output_root / 'img'

        # git_hash = subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode().strip()
        user = os.environ['LOGNAME'] if 'LOGNAME' in os.environ else os.environ.get('USERNAME', 'unknown')

        lightning_logger = None
        lightning_logger = MLFlowLogger(
            tracking_uri=config.get('mlflow_target_uri'),
            experiment_name=experiment_name,
            tags={MLFLOW_USER: user,
                  # MLFLOW_GIT_COMMIT: git_hash,
                  MLFLOW_RUN_NAME: config.get('run_name', 'default')},
            # tensorboard_path=checkpoint_path / 'tensorboard_log_dir'
        ) if config.get('mlflow_target_uri') is not None else None

        checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.img_dir.mkdir(exist_ok=True)
        # train_init_logging(run_output_root)
        shutil.copy2(args.config, run_output_root)

        if lightning_logger:
            lightning_logger.experiment.log_artifact(
                lightning_logger.run_id,
                str(run_output_root / args.config.name)
            )

    controller = DetectionController(config=config)

    trainer = configure_trainer(config, lightning_logger, checkpoint_path)

    if config.get('find_max_batch_size'):
        new_batch_size = find_max_batch_size(trainer, controller)
        if new_batch_size:
            config.train_batch_size = new_batch_size
            config.test_batch_size = new_batch_size
        controller = DetectionController(config=config)

    if config.get('find_optimal_init_lr'):
        new_lr = find_optimal_init_lr(trainer, controller)
        config.opt_params['main']['lr'] = new_lr
        controller = DetectionController(config=config)

    trainer.fit(controller)

    print('Completed!')
