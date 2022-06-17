from typing import Optional, List, Any

import torch
from pytorch_lightning.loops.dataloader import PredictionLoop as LegacyPredictionLoop
from pytorch_lightning.loops.epoch import PredictionEpochLoop as LegacyPredictionEpochLoop
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT


class PredictionEpochLoop(LegacyPredictionEpochLoop):

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        self.current_batch_indices = self._seen_batch_indices[batch_idx] if self._seen_batch_indices else []

        model_ref = self.trainer.lightning_module

        self.trainer.call_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_started()

        model_ref._current_fx_name = "predict_step"
        # predictions = self.trainer.accelerator.predict_step(step_kwargs)
        with self.trainer.accelerator.precision_plugin.predict_step_context():
            predictions = self.trainer.lightning_module.predict_step(*step_kwargs.values())

        self.batch_progress.increment_processed()

        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer.call_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_completed()

        if self.should_store_predictions:
            self.predictions.append(move_data_to_device(predictions, torch.device("cpu")))


class PredictionLoop(LegacyPredictionLoop):

    def __init__(self):
        super().__init__()
        self.predictions: Optional[List[List[Any]]] = None
        self.epoch_batch_indices: Optional[List[List[int]]] = None
        self.epoch_loop = PredictionEpochLoop()

        self._results = None  # for `trainer._results` access
        self._return_predictions: bool = False

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        results = self.predictions

        self.trainer.call_hook("on_predict_epoch_end", results)

        if self.return_predictions:
            return results
