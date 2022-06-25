from copy import deepcopy
from typing import Optional

import numpy as np
import pytorch_lightning
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.metrics import average_precision_score


class DetectionController(pytorch_lightning.LightningModule):
    logger: MLFlowLogger

    def __init__(self, config):
        super(DetectionController, self).__init__()
        self.config = config
        model = self.config.model()
        self.model_loss = self.config.loss(config, model)
        self.save_hyperparameters({i: repr(j) for i, j in config.items()})

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        with torch.no_grad():
            for i in range(len(batch[1])):
                batch[1][i]['labels'] = batch[1][i]['labels'] + 1
        loss = self.model_loss(*batch)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0) -> Optional[STEP_OUTPUT]:
        for i in range(len(batch[1])):
            batch[1][i]['labels'] = batch[1][i]['labels'] + 1
        out = self.model_loss(batch[0])
        return self.transfer_batch_to_device({'pred': out, 'true': batch[1]}, torch.device('cpu'), dataset_idx)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)
        exp_id = self.logger.run_id
        self.logger.experiment.log_artifacts(exp_id, str(self.config.output))

    def test_step(self, batch, batch_idx, dataset_idx=0) -> Optional[STEP_OUTPUT]:
        for i in range(len(batch[1])):
            batch[1][i]['labels'] = batch[1][i]['labels'] + 1
        out = self.model_loss(batch[0])
        return self.transfer_batch_to_device({'pred': out, 'true': batch[1]}, torch.device('cpu'), dataset_idx)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        thrs = (0.5, 0.7)
        for name, i in zip(('val', ), [1]):
            scores = [jj['scores'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            labels = [jj['labels'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            pred = [jj['boxes'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            target = [jj['boxes'].cpu().detach().numpy() for j in outputs[i] for jj in j['true']]
            target_labels = [jj['labels'].cpu().detach().numpy() for j in outputs[i] for jj in j['true']]

            iou = []

            for j in range(len(pred)):
                if len(pred[j]):
                    iou.append(self.intersection_over_union(np.round(pred[j][0]), target[j][0]))

            to_log = {
                f'{i} Mean IoU': np.mean(iou),
                f'{i} Median IoU': np.median(iou),
            }
            print(f'\n{name} Mean Detection IoU', np.mean(iou))
            # print(f'{name} Median IoU', np.median(iou))

            has_mask = 'masks' in outputs[i][0]['pred'][0] and 'masks' in outputs[i][0]['true'][0]
            if has_mask:
                pred_masks = [
                    (jj['masks'].cpu().detach().numpy() >= 0.49).astype(int) for j in outputs[i] for jj in j['pred']
                ]
                target_masks = [jj['masks'].cpu().detach().numpy().astype(int) for j in outputs[i] for jj in j['true']]
                masks_iou = [
                    ((pred_masks[j] == target_masks[j]) & (target_masks[j] == 1)).sum() /
                    ((pred_masks[j] == 1) | (target_masks[j] == 1)).sum()
                    for j in range(len(target_masks))
                ]
                masks_iou = self.av(masks_iou)
                to_log[f'Masks Mean IoU'] = masks_iou
                print(f'{name} Mean Segmentation IoU', masks_iou)

            preds_copy = pred
            scores_copy = scores
            target_copy = target

            for thr, thr_name in zip(thrs, (50, 70)):
                results = []
                pred = deepcopy(preds_copy)
                scores = deepcopy(scores_copy)
                target = deepcopy(target_copy)
                for j in range(len(pred)):
                    for a in range(len(pred[j])):
                        dt = pred[j][a]
                        results.append({'score': scores[j][a]})
                        ious = [
                            self.intersection_over_union(target[j][b], dt) for b in range(len(target[j]))
                            if labels[j][a] == target_labels[j][b]
                        ]
                        if len(ious) == 0:
                            max_IoU = -1
                            max_gt_id = -1
                        else:
                            max_gt_id, max_IoU = max([i for i in zip(list(range(len(ious))), ious)], key=lambda x: x[1])
                        if max_gt_id >= 0 and max_IoU >= thr:
                            results[-1]['TP'] = 1
                            target[j] = np.delete(target[j], max_gt_id, axis=0)
                        else:
                            results[-1]['TP'] = 0

                results = sorted(results, key=lambda k: k['score'], reverse=True)
                score = [i['score'] for i in results]
                flags = [i['TP'] for i in results]

                if len(flags) == 0:
                    mAP = 0.0
                else:
                    mAP = average_precision_score(flags, score)
                    # precision, recall, _ = precision_recall_curve(flags, score, pos_label=1)
                    # mAP = auc(recall, precision)
                to_log[f'AP {thr_name}'] = mAP
                print(f'{name} AP {thr_name}', mAP)

    def _evaluate(self, outputs: EPOCH_OUTPUT) -> None:
        thrs = (0.5, 0.7, 0.9)
        for name, i in zip(('train', 'val'), range(len(outputs))):
            scores = [jj['scores'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            labels = [jj['labels'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            pred = [jj['boxes'].cpu().detach().numpy() for j in outputs[i] for jj in j['pred']]
            target = [jj['boxes'].cpu().detach().numpy() for j in outputs[i] for jj in j['true']]
            target_labels = [jj['labels'].cpu().detach().numpy() for j in outputs[i] for jj in j['true']]


            iou = []

            for j in range(len(pred)):
                if len(pred[j]):
                    iou.append(self.intersection_over_union(np.round(pred[j][0]), target[j][0]))

            to_log = {
                f'{i} Mean IoU': np.mean(iou),
                f'{i} Median IoU': np.median(iou),
            }
            print(f'\n{name} Mean IoU', np.mean(iou))
            print(f'{name} Median IoU', np.median(iou))

            has_mask = 'masks' in outputs[i][0]['pred'][0] and 'masks' in outputs[i][0]['true'][0]
            if has_mask:
                pred_masks = [
                    (jj['masks'].cpu().detach().numpy() >= 0.5).astype(int) for j in outputs[i] for jj in j['pred']
                ]
                target_masks = [jj['masks'].cpu().detach().numpy().astype(int) for j in outputs[i] for jj in j['true']]
                masks_iou = [
                    ((pred_masks[j] == target_masks[j]) & (target_masks[j] == 1)).sum() /
                    ((pred_masks[j] == 1) | (target_masks[j] == 1)).sum()
                    for j in range(len(target_masks))
                ]
                masks_iou = self.av(masks_iou)
                to_log[f'Masks Mean IoU'] = masks_iou
                print(f'{name} Masks Mean IoU', masks_iou)

            preds_copy = pred
            scores_copy = scores
            target_copy = target

            for thr, thr_name in zip(thrs, (50, 70, 90)):
                results = []
                pred = deepcopy(preds_copy)
                scores = deepcopy(scores_copy)
                target = deepcopy(target_copy)
                for j in range(len(pred)):
                    for a in range(len(pred[j])):
                        dt = pred[j][a]
                        results.append({'score': scores[j][a]})
                        ious = [
                            self.intersection_over_union(target[j][b], dt) for b in range(len(target[j]))
                            if labels[j][a] == target_labels[j][b]
                        ]
                        if len(ious) == 0:
                            max_IoU = -1
                            max_gt_id = -1
                        else:
                            max_gt_id, max_IoU = max([i for i in zip(list(range(len(ious))), ious)], key=lambda x: x[1])
                        if max_gt_id >= 0 and max_IoU >= thr:
                            results[-1]['TP'] = 1
                            target[j] = np.delete(target[j], max_gt_id, axis=0)
                        else:
                            results[-1]['TP'] = 0

                results = sorted(results, key=lambda k: k['score'], reverse=True)
                score = [i['score'] for i in results]
                flags = [i['TP'] for i in results]

                if len(flags) == 0:
                    mAP = 0.0
                else:
                    mAP = average_precision_score(flags, score)
                    # precision, recall, _ = precision_recall_curve(flags, score, pos_label=1)
                    # mAP = auc(recall, precision)
                to_log[f'AP {thr_name}'] = mAP
                print(f'{name} AP {thr_name}', mAP)
            if self.logger is not None:
                self.logger.log_metrics({f'{name} {k}': v for k, v in to_log.items()}, self.current_epoch)

    @staticmethod
    def intersection_over_union(dt_bbox, gt_bbox):
        x0 = max(dt_bbox[0], gt_bbox[0])
        x1 = min(dt_bbox[2], gt_bbox[2])
        y0 = max(dt_bbox[1], gt_bbox[1])
        y1 = min(dt_bbox[3], gt_bbox[3])
        intersection = (x1 - x0) * (y1 - y0)
        union = (
                (dt_bbox[2] - dt_bbox[0]) * (dt_bbox[3] - dt_bbox[1]) +
                (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]) -
                intersection
        )
        iou = intersection / union
        return iou

    @staticmethod
    def av(l):
        return np.mean([i for i in l if not np.isnan(i)])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.config.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = self.config.get('test_dataloader')
        if dl is not None:
            return dl()
        return self.config.val_dataloader()

    def configure_optimizers(self):
        return self.config.optimizer(self.model_loss)


class YOLOV4DetectionController(DetectionController):

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch[0]
        bboxes = [i['boxes'] for i in batch[1]]
        loss = self.model_loss(images, bboxes)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx) -> Optional[STEP_OUTPUT]:
        out = self.model_loss(batch[0])
        return {'pred': out, 'true': batch[1]}
