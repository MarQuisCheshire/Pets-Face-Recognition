import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from engine import Controller
from preprocessor import Preproc3, Preproc4
from utils import get_dict_wrapper, configure_trainer


class ToNumpy(torch.nn.Module):

    def forward(self, x):
        return x.cpu().detach().numpy()


@torch.no_grad()
def calibration(model: Controller, tag: str):
    calibrator = IsotonicRegression(y_min=0., y_max=1.)
    Path('results/tmp').mkdir(exist_ok=True, parents=True)

    trainer = configure_trainer(model.config, False, Path('results/tmp'))
    predictions = trainer.predict(model)

    emb = predictions[0]['emb']
    classes = predictions[0]['label']
    indices = predictions[0]['index']
    s = torch.argsort(indices)
    emb = emb[s]
    classes = classes[s]
    name, pair_generator = model.config.pair_generator(0)
    similarity_f = model.config.similarity_f
    scores = similarity_f([(emb[id1], emb[id2]) for id1, id2 in pair_generator.corrected_indices])
    labels = torch.as_tensor(pair_generator.labels)
    scores = scores.cpu()

    X = scores.cpu().detach().numpy()
    y = labels.cpu().detach().numpy()
    calibrator.fit(X, y)
    print(calibrator.__dict__)
    with open(f'calibrator_{tag}.pickle', 'wb') as f:
        pickle.dump(calibrator, f)


@torch.no_grad()
def main():
    device = 'cuda:0'

    body_preproc = Preproc4(device=device, thr=0.7, mask_thr=True)

    head_preproc = Preproc3(
        np.array([[70, 92], [154, 92], [112, 160]]),
        (224, 224, 3),
        (0, 0, 0),
        device=device
    )

    dog_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/1/f6fca5573f62410ab0649407c951c153/artifacts/checkpoints'
                '/1/f6fca5573f62410ab0649407c951c153/checkpoints/epoch=36-step=42142.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/1/f6fca5573f62410ab0649407c951c153/artifacts/fe_dogs_config.py')
    ).eval()

    cat_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/10/cfc4f63a71c2417087adb2d3feb5d34c/artifacts/checkpoints'
                '/10/cfc4f63a71c2417087adb2d3feb5d34c/checkpoints/epoch=42-step=51943.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/10/cfc4f63a71c2417087adb2d3feb5d34c/artifacts/cat_fe_head.py')
    ).eval()
    # cat_model = Controller.load_from_checkpoint(
    #     str(
    #         Path(
    #             'mlruns/10/bdf632f3e10b49a6b27457f72abb6a2e/artifacts/checkpoints'
    #             '/10/bdf632f3e10b49a6b27457f72abb6a2e/checkpoints/epoch=11-step=14495.ckpt'
    #         )
    #     ),
    #     config=get_dict_wrapper('mlruns/10/bdf632f3e10b49a6b27457f72abb6a2e/artifacts/cat_fe_head.py')
    # ).eval().model_loss
    dog_model.add_margin = None
    cat_model.add_margin = None
    dog_model.to(device)
    cat_model.to(device)

    # dog_body_model = Controller.load_from_checkpoint(
    #     str(
    #         Path(
    #             'mlruns/1/f6fca5573f62410ab0649407c951c153/artifacts/checkpoints'
    #             '/1/f6fca5573f62410ab0649407c951c153/checkpoints/epoch=36-step=42142.ckpt'
    #         )
    #     ),
    #     config=get_dict_wrapper('mlruns/11/84ff365352fa4e058b30c882bc00a607/artifacts/body_dog_fe.py')
    # ).eval().model_loss

    dog_body_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/11/f19ad652be834f618261844c9d63927a/artifacts/checkpoints'
                '/11/f19ad652be834f618261844c9d63927a/checkpoints/epoch=37-step=96671.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/11/f19ad652be834f618261844c9d63927a/artifacts/body_dog_fe.py')
    ).eval()

    # cat_body_model = Controller.load_from_checkpoint(
    #     str(
    #         Path(
    #             'mlruns/11/84ff365352fa4e058b30c882bc00a607/artifacts/checkpoints'
    #             '/11/84ff365352fa4e058b30c882bc00a607/checkpoints/epoch=39-step=100559.ckpt'
    #         )
    #     ),
    #     config=get_dict_wrapper('mlruns/13/6502b10363974f0f825e709b522ee659/artifacts/body_cat_fe.py')
    # ).eval().model_loss
    # cat_body_model = Controller.load_from_checkpoint(
    #     str(
    #         Path(
    #             'mlruns/13/7c1fc7d0fae74936b9297c6669335aa0/artifacts/checkpoints'
    #             '/13/7c1fc7d0fae74936b9297c6669335aa0/checkpoints/epoch=6-step=14244.ckpt'
    #         )
    #     ),
    #     config=get_dict_wrapper('mlruns/13/7c1fc7d0fae74936b9297c6669335aa0/artifacts/body_cat_fe.py')
    # ).eval().model_loss
    cat_body_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/13/e8070f822d8b4b31b7086c1e3add5dca/artifacts/checkpoints'
                '/13/e8070f822d8b4b31b7086c1e3add5dca/checkpoints/epoch=8-step=18314.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/13/e8070f822d8b4b31b7086c1e3add5dca/artifacts/body_cat_fe.py')
    ).eval()
    dog_body_model.add_margin = None
    cat_body_model.add_margin = None
    dog_body_model.to(device)
    cat_body_model.to(device)

    calibration(cat_model, 'cat_model')
    calibration(dog_model, 'dog_model')
    calibration(cat_body_model, 'cat_body_model')
    calibration(dog_body_model, 'dog_body_model')


if __name__ == '__main__':
    main()
