import logging
import warnings
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Iterable
from typing import Union, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loops import TrainingBatchLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.plugins import PLUGIN_INPUT, TrainingTypePlugin, DDPSpawnPlugin
from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.states import TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import parsing, GradClipAlgorithmType, rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT

from .loops import EvaluationLoop, TrainingEpochLoop, PredictionLoop


def mlflow_experiment_exit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        r = func(*args, **kwargs)
        if args[0].logger is not None:
            args[0].logger.finalize('FINISHED')
        return r
        # try:
        #     r = func(*args, **kwargs)
        #     if args[0].logger is not None:
        #         args[0].logger.finalize('FINISHED')
        #     return r
        # except KeyboardInterrupt:
        #     if args[0].logger is not None:
        #         args[0].logger.finalize('FINISHED')
        # except Exception as error:
        #     if args[0].logger is not None:
        #         args[0].logger.finalize('FAILED')
        #     raise Exception('FAILED\t' * 5) from error

    return inner


log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class Trainer(pl.Trainer):
    @_defaults_from_env_vars
    def __init__(
            self,
            logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
            checkpoint_callback: Optional[bool] = None,
            enable_checkpointing: bool = False,
            callbacks: Optional[Union[List[Callback], Callback]] = None,
            default_root_dir: Optional[str] = None,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None,
            process_position: int = 0,
            num_nodes: int = 1,
            num_processes: int = 1,
            devices: Optional[Union[List[int], str, int]] = None,
            gpus: Optional[Union[List[int], str, int]] = None,
            auto_select_gpus: bool = False,
            tpu_cores: Optional[Union[List[int], str, int]] = None,
            ipus: Optional[int] = None,
            log_gpu_memory: Optional[str] = None,
            progress_bar_refresh_rate: Optional[int] = None,
            enable_progress_bar: bool = True,
            overfit_batches: Union[int, float] = 0.0,
            track_grad_norm: Union[int, float, str] = -1,
            check_val_every_n_epoch: int = 1,
            fast_dev_run: Union[int, bool] = False,
            accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
            max_epochs: Optional[int] = None,
            min_epochs: Optional[int] = None,
            max_steps: int = -1,
            min_steps: Optional[int] = None,
            max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
            limit_train_batches: Union[int, float] = 1.0,
            limit_val_batches: Union[int, float] = 1.0,
            limit_test_batches: Union[int, float] = 1.0,
            limit_predict_batches: Union[int, float] = 1.0,
            val_check_interval: Union[int, float] = 1.0,
            flush_logs_every_n_steps: Optional[int] = None,
            log_every_n_steps: int = 50,
            accelerator: Optional[Union[str, Accelerator]] = None,
            strategy: Optional[Union[str, TrainingTypePlugin]] = None,
            sync_batchnorm: bool = False,
            precision: Union[int, str] = 32,
            enable_model_summary: bool = True,
            weights_summary: Optional[str] = "top",
            weights_save_path: Optional[str] = None,
            num_sanity_val_steps: int = 0,
            resume_from_checkpoint: Optional[Union[Path, str]] = None,
            profiler: Optional[Union[BaseProfiler, str]] = None,
            benchmark: bool = False,
            deterministic: bool = False,
            reload_dataloaders_every_n_epochs: int = 0,
            reload_dataloaders_every_epoch: bool = False,
            auto_lr_find: Union[bool, str] = False,
            replace_sampler_ddp: bool = False,
            detect_anomaly: bool = False,
            auto_scale_batch_size: Union[str, bool] = False,
            prepare_data_per_node: Optional[bool] = None,
            plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
            amp_backend: str = "native",
            amp_level: Optional[str] = None,
            move_metrics_to_cpu: bool = False,
            multiple_trainloader_mode: str = "max_size_cycle",
            stochastic_weight_avg: bool = False,
            terminate_on_nan: Optional[bool] = None,
    ):
        r"""
        Customize every aspect of training via flags.

        Args:

            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")
                as well as custom accelerator instances.

                .. deprecated:: v1.5
                    Passing training strategies (e.g., 'ddp') to ``accelerator`` has been deprecated in v1.5.0
                    and will be removed in v1.7.0. Please use the ``strategy`` argument instead.

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            amp_backend: The mixed precision backend to use ("native" or "apex").

            amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
                if ``amp_backend`` is set to "apex".

            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.

            auto_select_gpus: If enabled and ``gpus`` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            benchmark: If true enables cudnn.benchmark.

            callbacks: Add a callback or list of callbacks.

            checkpoint_callback: If ``True``, enable checkpointing.

                .. deprecated:: v1.5
                    ``checkpoint_callback`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please consider using ``enable_checkpointing`` instead.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.

            check_val_every_n_epoch: Check val every n train epochs.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            detect_anomaly: Enable anomaly detection for the autograd engine.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Default: ``False``.

            devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                based on the accelerator type.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).

            flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

                .. deprecated:: v1.5
                    ``flush_logs_every_n_steps`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please configure flushing directly in the logger instead.

            gpus: Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
                provided and the `save_dir` property of that logger is not set, local files (checkpoints,
                profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
                of the individual loggers.

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance.

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please use the ``DeviceStatsMonitor`` callback directly instead.

            log_every_n_steps: How often to log within steps (defaults to every 50 steps).

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please set ``prepare_data_per_node`` in LightningDataModule or LightningModule directly instead.

            process_position: Orders the progress bar when running multiple models on same machine.

                .. deprecated:: v1.5
                    ``process_position`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``process_position``
                    directly to the Trainer's ``callbacks`` argument instead.

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
                a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

                .. deprecated:: v1.5
                    ``progress_bar_refresh_rate`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``refresh_rate``
                    directly to the Trainer's ``callbacks`` argument instead. To disable the progress bar,
                    pass ``enable_progress_bar = False`` to the Trainer.

            enable_progress_bar: Whether to enable to progress bar by default.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.

            overfit_batches: Overfit a fraction of training data (float) or a set number of batches (int).

            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.

            precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
                Can be used on CPU, GPU or TPUs.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).
                If both min_epochs and min_steps are not specified, defaults to ``min_epochs = 1``.

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            max_time: Stop training after this amount of time has passed. Disabled by default (None).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            num_nodes: Number of GPU nodes for distributed training.

            num_processes: Number of processes for distributed training with ``accelerator="cpu"``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.

            reload_dataloaders_every_n_epochs: Set to a non-negative integer to reload dataloaders every n epochs.

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

                .. deprecated:: v1.4
                    ``reload_dataloaders_every_epoch`` has been deprecated in v1.4 and will be removed in v1.6.
                    Please use ``reload_dataloaders_every_n_epochs``.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

                .. deprecated:: v1.5
                    ``resume_from_checkpoint`` is deprecated in v1.5 and will be removed in v1.7.
                    Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead.

            strategy: Supports different training strategies with aliases
                as well custom training type plugins.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

                .. deprecated:: v1.5
                    Trainer argument ``terminate_on_nan`` was deprecated in v1.5 and will be removed in 1.7.
                    Please use ``detect_anomaly`` instead.

            detect_anomaly: Enable anomaly detection for the autograd engine.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            ipus: How many IPUs to train on.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using
                Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.

            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).

            enable_model_summary: Whether to enable model_ summarization by default.

            weights_summary: Prints a summary of the weights when training begins.

                .. deprecated:: v1.5
                    ``weights_summary`` has been deprecated in v1.5 and will be removed in v1.7.
                    To disable the summary, pass ``enable_model_summary = False`` to the Trainer.
                    To customize the summary, pass :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary`
                    directly to the Trainer's ``callbacks`` argument.

            weights_save_path: Where to save weights if specified. Will override default_root_dir
                for checkpoints only. Use this if for whatever reason you need the checkpoints
                stored in a different place than the logs written in `default_root_dir`.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                Defaults to `default_root_dir`.

            move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
                This can save some gpu memory, but can make training slower. Use with attention.

            multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
                In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
                and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
                reload when reaching the minimum length of datasets.

            stochastic_weight_avg: Whether to use `Stochastic Weight Averaging (SWA)
                <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>`_.

                .. deprecated:: v1.5
                    ``stochastic_weight_avg`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging`
                    directly to the Trainer's ``callbacks`` argument instead.
        """
        Trainer._log_api_event("init")
        self.state = TrainerState()

        gpu_ids, tpu_cores = self._parse_devices(gpus, auto_select_gpus, tpu_cores)

        # init connectors
        self._data_connector = DataConnector(self, multiple_trainloader_mode)

        self._accelerator_connector = AcceleratorConnector(
            num_processes,
            devices,
            tpu_cores,
            ipus,
            accelerator,
            strategy,
            gpus,
            gpu_ids,
            num_nodes,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp,
            deterministic,
            precision,
            amp_backend,
            amp_level,
            plugins,
        )
        self.logger_connector = LoggerConnector(self, log_gpu_memory)
        self._callback_connector = CallbackConnector(self)
        self.checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
        self.signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

        fit_loop = FitLoop(
            min_epochs=(1 if (min_epochs is None and min_steps is None and max_time is None) else min_epochs),
            max_epochs=(
                max_epochs if max_epochs is not None else (1000 if (max_steps == -1 and max_time is None) else -1)
            ),
        )
        training_epoch_loop = TrainingEpochLoop(min_steps, max_steps)
        training_batch_loop = TrainingBatchLoop()
        training_validation_loop = EvaluationLoop()
        training_epoch_loop.connect(batch_loop=training_batch_loop, val_loop=training_validation_loop)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # default .fit() loop
        self.fit_loop = fit_loop

        # default .validate() loop
        self.validate_loop = EvaluationLoop()

        # default .test() loop
        self.test_loop = EvaluationLoop()

        # default .predict() loop
        self.predict_loop = PredictionLoop()

        # Needed because of LightningOptimizer
        self._lightning_optimizers = None

        # .validate() and .test() set this when they load a checkpoint
        self.validated_ckpt_path: Optional[str] = None
        self.tested_ckpt_path: Optional[str] = None
        self.predicted_ckpt_path: Optional[str] = None

        # todo: remove in v1.7
        self._weights_summary: Optional[str] = None

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            checkpoint_callback,
            enable_checkpointing,
            enable_progress_bar,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            enable_model_summary,
            weights_summary,
            stochastic_weight_avg,
            max_time,
            accumulate_grad_batches,
        )

        # hook
        self.on_init_start()

        # init optimizer + lr scheduler related flags
        self.lr_schedulers = []
        self.optimizers = []
        self.optimizer_frequencies = []

        # init data flags
        self._data_connector.on_trainer_init(
            check_val_every_n_epoch,
            reload_dataloaders_every_n_epochs,
            reload_dataloaders_every_epoch,
            prepare_data_per_node,
        )

        if terminate_on_nan is not None:
            rank_zero_deprecation(
                "Trainer argument `terminate_on_nan` was deprecated in v1.5 and will be removed in 1.7."
                " Please use `Trainer(detect_anomaly=True)` instead."
            )
            if not isinstance(terminate_on_nan, bool):
                raise TypeError(f"`terminate_on_nan` should be a bool, got {terminate_on_nan}.")

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
                gradient_clip_algorithm.lower()
        ):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        # gradient norm tracking
        if track_grad_norm != -1 and not (
                (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
        ):
            raise MisconfigurationException(
                f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self._terminate_on_nan = terminate_on_nan
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower())
            if gradient_clip_algorithm is not None
            else gradient_clip_algorithm
        )
        self.track_grad_norm: float = float(track_grad_norm)

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init(num_sanity_val_steps)

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.__init_profiler(profiler)

        # init logger flags
        self.logger: Optional[LightningLoggerBase]
        self.logger_connector.on_trainer_init(logger, flush_logs_every_n_steps, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self._init_debugging_flags(
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            val_check_interval,
            overfit_batches,
            fast_dev_run,
        )

        # Callback system
        self.on_init_end()

    @mlflow_experiment_exit
    def _run(
            self, model: "pl.LightningModule", ckpt_path: Optional[str] = None
    ) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # model.check_resume(self.testing)

        # attach model_ to the training type plugin
        self.training_type_plugin.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        verify_loop_configurations(self)

        # hook
        self._data_connector.prepare_data()

        # ----------------------------
        # SET UP TRAINING
        # ----------------------------
        self.call_hook("on_before_accelerator_backend_setup")
        self.accelerator.setup_environment()
        self._call_setup_hook()  # allow user to setup lightning_module in accelerator environment

        # check if we should delay restoring checkpoint till later
        if not self.training_type_plugin.restore_checkpoint_after_pre_dispatch:
            self._restore_modules_and_callbacks(ckpt_path)

        self._call_configure_sharded_model()  # allow user to setup in model_ sharded environment
        self.accelerator.setup(self)

        # ----------------------------
        # INSPECT THE CORE LOOPS
        # ----------------------------
        fr"""
             Lightning internal flow looks like this:
        {Trainer.fit} or {Trainer.test} or {Trainer.predict}  ||
                                |                             ||
                        create accelerator                    ||
                                |                             ||
                         {self._dispatch}                     ||
                                |                             ||  LIGHTNING
         {self.training_type_plugin.start_training}           ||
       or {self.training_type_plugin.start_evaluating}        ||
       or {self.training_type_plugin.start_predicting}        ||  FLOW
                                |                             ||
                         {self.run_stage}                     ||
                                |                             ||  DIRECTION
                        {self._run_train}                     ||
                     or {self._run_evaluate}                  ||
                     or {self._run_predict}                   ||
                                |                             ||
                             results                          \/
        This is used to guide readers to the core loops: train, test, predict.
        {self._run_predict} is the simplest to understand, use `Go to Definition` to read it :)
        Search for `start_training` or `start_evaluating` or `start_predicting` in
        `pytorch_lightning/plugins/training_type_plugin` to find accelerator dispatch functions.
        """

        # ----------------------------
        # TRAIN
        # ----------------------------

        # reset logger connector
        self.logger_connector.reset_results()
        self.logger_connector.reset_metrics()

        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook("on_fit_start")

        # plugin will setup fitting (e.g. ddp will launch child processes)
        self._pre_dispatch()

        if self.training_type_plugin.restore_checkpoint_after_pre_dispatch:
            self._restore_modules_and_callbacks(ckpt_path)

        # restore optimizers, etc.
        self.checkpoint_connector.restore_training_state()

        self.checkpoint_connector.resume_end()

        # dispatch `start_training` or `start_evaluating` or `start_predicting`
        self._dispatch()

        # plugin will finalized fitting (e.g. ddp_spawn will load trained model_)
        self._post_dispatch()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook("on_fit_end")

        # teardown if necessary (similar calls for spawn plugins are excluded as they have
        # been included at the end of `new_process` functions)
        if not isinstance(self.training_type_plugin, DDPSpawnPlugin):
            self._call_teardown_hook()

        if self.state.status != TrainerStatus.INTERRUPTED:
            self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return self.training_type_plugin.results

    @property
    def is_distributed_run(self) -> bool:
        return self._accelerator_connector.is_distributed

    @property
    def sync_batchnorm_flag(self) -> bool:
        return self._accelerator_connector.sync_batchnorm
