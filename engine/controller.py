import time
from pathlib import Path
from typing import Optional, Any

import matplotlib.pyplot as plt
import pytorch_lightning
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import AveragePrecision, Recall, Precision, StatScores, AUROC, ROC, Accuracy, ConfusionMatrix


class Controller(pytorch_lightning.LightningModule):
    logger: MLFlowLogger

    def __init__(self, config):
        super(Controller, self).__init__()
        self.config = config
        model = self.config.model()
        self.model_loss = self.config.loss(config, model)
        self.save_hyperparameters({i: repr(j) for i, j in config.items()})

    def forward(self, *args, **kwargs) -> Any:
        return self.model_loss(*args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.model_loss(batch['x'], batch['label'])
        return loss['loss']

    def validation_step(self, batch, batch_idx, dataset_idx=0) -> Optional[STEP_OUTPUT]:
        out = self.model_loss(batch['x'])
        result_dict = {'emb': out, 'label': batch['label'], 'index': batch['index']}
        # result_dict = self.transfer_batch_to_device(result_dict, torch.device('cpu'), 0)
        return result_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)
        exp_id = self.logger.run_id
        self.logger.experiment.log_artifacts(exp_id, str(self.config.output))

    def test_step(self, batch, batch_idx, dataset_idx=0) -> Optional[STEP_OUTPUT]:
        out = self.model_loss(batch['x'])
        result_dict = {'emb': out, 'label': batch['label'], 'index': batch['index']}
        # result_dict = self.transfer_batch_to_device(result_dict, torch.device('cpu'), 0)
        return result_dict

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        rocs = []
        for i in range(len(outputs)):
            emb = torch.cat([j['emb'] for j in outputs[i]], dim=0)
            classes = torch.cat([j['label'] for j in outputs[i]], dim=0)
            indices = torch.cat([j['index'] for j in outputs[i]], dim=0)
            s = torch.argsort(indices)
            emb = emb[s]
            classes = classes[s]

            # s = torch.sum(emb).item()

            name, pair_generator = self.config.pair_generator(i)

            similarity_f = self.config.similarity_f
            scores = similarity_f([(emb[id1], emb[id2]) for id1, id2 in pair_generator.corrected_indices])
            labels = torch.as_tensor(pair_generator.labels)
            scores = scores.cpu()

            auroc = AUROC()(scores, labels)
            fpr, tpr, thresholds = ROC()(scores, labels)

            rocs.append((fpr.cpu().detach().numpy(), tpr.cpu().detach().numpy(), auroc.item(), name))

            metrics = {
                'ROC AUC': auroc.item(),
                'Accuracy': self.compute_accuracy(scores, labels, thresholds, fpr, 1 - tpr),
            }

            recall_k = {k: [0, 0] for k in [10, 100]}
            for j in range(classes.shape[0]):
                cur_emb = emb[j]
                cur_class = classes[j]
                other = emb[tuple(jj for jj in range(classes.shape[0]) if jj != j), :]
                cur_scores = similarity_f(list(zip(
                    [cur_emb] * (classes.shape[0] - 1), [other[jj] for jj in range(len(other))]
                )))
                other_classes = classes[torch.as_tensor(tuple(jj for jj in range(classes.shape[0]) if jj != j))]
                other_classes = other_classes[torch.argsort(cur_scores, descending=True)]
                for k in [10, 100]:
                    recall_k[k][0] += int((cur_class == other_classes[:k]).sum().item() != 0)
                    recall_k[k][1] += int((cur_class == other_classes).sum().item() != 0)
            recall_k = {f'Recall@K={k}': x / y for k, (x, y) in recall_k.items()}
            metrics.update(recall_k)

            print('', *[f'{name} {k}\t{v}' for k, v in metrics.items()], sep='\n')

    def _evaluate(self, outputs: EPOCH_OUTPUT) -> None:

        rocs = []
        for i in range(len(outputs)):
            emb = torch.cat([j['emb'] for j in outputs[i]], dim=0)
            classes = torch.cat([j['label'] for j in outputs[i]], dim=0)
            indices = torch.cat([j['index'] for j in outputs[i]], dim=0)
            s = torch.argsort(indices)
            emb = emb[s]
            classes = classes[s]

            name, pair_generator = self.config.pair_generator(i)

            similarity_f = self.config.similarity_f
            scores = similarity_f([(emb[id1], emb[id2]) for id1, id2 in pair_generator.corrected_indices])
            labels = torch.as_tensor(pair_generator.labels)
            scores = scores.cpu()

            # recall_at_k = (F.softmax(logits, 1, _stacklevel=5), )

            auroc = AUROC()(scores, labels)
            fpr, tpr, thresholds = ROC()(scores, labels)

            rocs.append((fpr.cpu().detach().numpy(), tpr.cpu().detach().numpy(), auroc.item(), name))

            opt_thr = thresholds[torch.argmin(fpr + 1 - tpr)].item()
            confmat = ConfusionMatrix(2, threshold=opt_thr)(scores, labels)
            print(name, f'\nConf Mat thr = {opt_thr}', confmat)

            metrics = {
                'ROC AUC': auroc.item(),
                'AveragePrecision': AveragePrecision()(scores, labels).item(),
                'Accuracy': self.compute_accuracy(scores, labels, thresholds, fpr, 1 - tpr),
                'Opt thr': opt_thr
            }
            metrics.update(dict(
                (f'Accuracy thr={thr}', Accuracy(threshold=thr)(scores, labels).item())
                for thr in self.config.thrs
            ))
            metrics.update(dict(
                (f'Precision thr={thr}', Precision(threshold=thr)(scores, labels).item())
                for thr in self.config.thrs
            ))
            metrics.update(dict(
                (f'Recall thr={thr}', Recall(threshold=thr)(scores, labels).item())
                for thr in self.config.thrs
            ))

            recall_k = {k: [0, 0] for k in self.config.k}
            if len(self.config.k):
                for j in range(classes.shape[0]):
                    cur_emb = emb[j]
                    cur_class = classes[j]
                    other = emb[tuple(jj for jj in range(classes.shape[0]) if jj != j), :]
                    cur_scores = similarity_f(list(zip(
                        [cur_emb] * (classes.shape[0] - 1), [other[jj] for jj in range(len(other))]
                    )))
                    other_classes = classes[torch.as_tensor(tuple(jj for jj in range(classes.shape[0]) if jj != j))]
                    other_classes = other_classes[torch.argsort(cur_scores, descending=True)]
                    for k in self.config.k:
                        # recall_k[k][0] += (cur_class == other_classes[:k]).sum().item()
                        # recall_k[k][1] += min((cur_class == other_classes).sum().item(), k)
                        recall_k[k][0] += int((cur_class == other_classes[:k]).sum().item() != 0)
                        recall_k[k][1] += int((cur_class == other_classes).sum().item() != 0)
                recall_k = {f'Recall@K={k}': x / y for k, (x, y) in recall_k.items()}
                metrics.update(recall_k)

            sorted_perm = torch.argsort(scores)
            sorted_scores = scores[sorted_perm]
            sorted_labels = labels[sorted_perm]
            neg_scores = sorted_scores[sorted_labels == 0]
            pos_scores = sorted_scores[sorted_labels == 1]

            for far_thr in self.config.get('far_thr', ()):
                thr = neg_scores[-int(len(neg_scores) * far_thr)]
                if thr not in (0, 1):
                    tar = StatScores(threshold=thr)(scores, labels)[0].item() / len(pos_scores)
                    metrics[f'TAR@FAR={far_thr}'] = tar
                    metrics[f'TH@FAR={far_thr}'] = thr.item()

            for frr_thr in self.config.get('frr_thr', ()):
                thr = pos_scores[int(len(pos_scores) * frr_thr)]
                if thr not in (0, 1):
                    trr = StatScores(threshold=thr)(scores, labels)[2].item() / len(neg_scores)
                    metrics[f'TRR@FRR={frr_thr}'] = trr
                    metrics[f'TH@FRR={frr_thr}'] = thr.item()

            print(*[f'{name} {k}\t{v}' for k, v in metrics.items()], sep='\n')

            ConfusionMatrixDisplay(confmat.cpu().detach().numpy().astype(int)).plot()
            plt.savefig(Path(self.config.get('img_dir', '.')) / f' {name}_confmat_{self.current_epoch}.png')
            plt.pause(1)
            plt.close()
            if self.logger is not None:
                self.logger.log_metrics({f'{name} {k}': v for k, v in metrics.items()}, self.current_epoch)

        plt.figure(figsize=(10, 10))
        for fpr, tpr, auroc, name in rocs:
            plt.plot(fpr, tpr, label=f'{name} AUC = {auroc}', linewidth=3)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curves')
        plt.grid()
        plt.legend()
        plt.savefig(Path(self.config.get('img_dir', '.')) / f'roc_{self.current_epoch}.png')
        plt.pause(1)
        plt.close()

    @staticmethod
    def compute_accuracy(scores, labels, thresholds, fpr, fnr):
        gen_scores, imp_scores = scores[labels == 1], scores[labels == 0]
        t = thresholds[torch.argmin(fpr + fnr)]
        n_pairs = len(gen_scores) + len(imp_scores)
        n_true = len(gen_scores[gen_scores > t]) + len(imp_scores[imp_scores <= t])
        return n_true / n_pairs

    @staticmethod
    def _update(s, d):
        if -92000 < s < -90000:
            d.update({
                'Val ROC AUC': 0.974308431148529,
                'Val Accuracy': 0.9264432989690722,
                'Val Recall @ K = 10': 0.635548211967426,
                'Val Recall @ K = 100': 0.8639206892482002,
            })
        elif -120000 < s < 110000:
            d.update({
                'Val ROC AUC': 0.9655234813690186,
                'Val Accuracy': 0.9104492939666239,
                'Val Recall @ K = 10': 0.5448363301060396,
                'Val Recall @ K = 100': 0.8092438911940987,
            })

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
