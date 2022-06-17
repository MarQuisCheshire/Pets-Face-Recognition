from pytorch_lightning.loops.epoch import TrainingEpochLoop as LegacyTrainingEpochLoop


class TrainingEpochLoop(LegacyTrainingEpochLoop):

    def on_advance_end(self):
        # -----------------------------------------
        # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self._run_validation()
            self.trainer.training = True

            if self.trainer.is_distributed_run:
                self.trainer.training_type_plugin.barrier()

        # -----------------------------------------
        # SAVE LOGGERS (ie: Tensorboard, etc...)
        # -----------------------------------------
        self._save_loggers_on_train_batch_end()

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # progress global step according to grads progress
            self.global_step += 1

        # if training finished, try to exit in `on_run_end` instead as we should have enough time
        # TODO: @tchaton verify this assumption is True.
        if not self._is_training_done:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

