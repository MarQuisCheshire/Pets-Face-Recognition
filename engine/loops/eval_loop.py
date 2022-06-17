import logging
from typing import Any, List

from deprecate.utils import void
from pytorch_lightning.loops.dataloader import EvaluationLoop as LegacyEvaluationLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT


class EvaluationLoop(LegacyEvaluationLoop):

    def advance(self, *args: Any, **kwargs: Any) -> None:
        void(*args, **kwargs)

        dataloader_idx: int = self.current_dataloader_idx
        dataloader = self.trainer.training_type_plugin.process_dataloader(self.current_dataloader)
        self.data_fetcher = dataloader = self.trainer._data_connector.get_profiled_dataloader(
            dataloader, dataloader_idx=dataloader_idx
        )

        dl_max_batches = self._max_batches[dataloader_idx]
        dl_outputs = self.epoch_loop.run(dataloader, dataloader_idx, dl_max_batches, self.num_dataloaders)

        # store batch level output per dataloader
        self.outputs.append(dl_outputs)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

    def on_run_end(self) -> List[_OUT_DICT]:
        outputs = self.outputs

        # free memory
        self.outputs = []

        # lightning module method
        self._evaluation_epoch_end(outputs)

        # hook
        self._on_evaluation_epoch_end()

        # log epoch metrics
        eval_loop_results = self.trainer.logger_connector.update_eval_epoch_metrics()

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        return eval_loop_results
