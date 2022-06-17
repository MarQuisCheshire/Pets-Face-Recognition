import json
import pickle
from contextlib import suppress
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from engine import Controller
from preprocessor import Preproc3, Preproc4
from utils import get_dict_wrapper
from utils.preprocs import resize_with_padding


def process_base(
        base: Path,
        head_pipeline: Optional[Callable],
        body_pipeline: Optional[Callable]
) -> Dict[Path, Dict[str, Any]]:
    base_dict = {}
    for folder in tqdm(list(base.iterdir())):
        images_paths = [i for i in folder.iterdir() if i.name != 'card.json']
        with open(folder / 'card.json', 'r') as f:
            type_ = int(json.load(f)['animal'])

        vectors = [head_pipeline(np.array(Image.open(i).convert('RGB')), type_) for i in images_paths]
        vectors = [i for i in vectors if i is not None]
        body_vectors = [body_pipeline(np.array(Image.open(i).convert('RGB')), type_) for i in images_paths]
        body_vectors = [i for i in body_vectors if i is not None]
        if len(vectors) != 0 or len(body_vectors) != 0:
            base_dict[folder.resolve()] = {
                'type': type_,
                'head_vectors': vectors,
                'body_vectors': body_vectors
            }
        # else:
        #     print('Lost', folder.resolve())
    print(len(base_dict))
    return base_dict


def prepare_data(path: Path, head_pipeline: Optional[Callable], body_pipeline: Optional[Callable]) -> Dict[Path, Any]:
    assert (path / 'found').exists() and (path / 'lost').exists()

    db = {}
    for big_folder in ((path / 'found').resolve(), (path / 'lost').resolve()):
        initial_base = big_folder / str(big_folder.name)
        extra_base = [i for i in big_folder.iterdir() if i.resolve() != initial_base][0]

        initial_base_dict = process_base(initial_base, head_pipeline, body_pipeline)
        extra_base_dict = process_base(extra_base, head_pipeline, body_pipeline)

        db[big_folder.resolve()] = initial_base_dict, extra_base_dict

    return db


def similarity_f(pairs):
    t1 = torch.cat([i[0].unsqueeze(0) for i in pairs], dim=0)
    t2 = torch.cat([i[1].unsqueeze(0) for i in pairs], dim=0)

    return (F.cosine_similarity(t1, t2) + 1) / 2


def mean_strategy_cal_scores(v1: List[torch.Tensor], v2: List[torch.Tensor]) -> float:
    pairs = []
    for i in v1:
        for j in v2:
            pairs.append((i, j))

    scores = similarity_f(pairs)
    return torch.mean(scores).clamp(min=0.0).item()


def max_strategy_cal_scores(v1: List[torch.Tensor], v2: List[torch.Tensor]) -> float:
    pairs = []
    for i in v1:
        for j in v2:
            pairs.append((i, j))

    scores = similarity_f(pairs)
    return torch.max(scores).item()


def calc_scores(init_db: Dict[Path, Any], extra_db: Dict[Path, Any]) -> List[Any]:
    df = []
    for f, enroll in tqdm(init_db.items(), total=len(init_db)):
        v1 = enroll['head_vectors']
        v1_body = enroll['body_vectors']
        type_ = enroll['type']

        l = []
        for f2, verify in extra_db.items():
            if verify['type'] != type_:
                continue
            score = {0: 0, 1: 0}
            if len(v1) != 0 and len(verify['head_vectors']) != 0:
                score[0] = mean_strategy_cal_scores(v1, verify['head_vectors'])
            if len(v1_body) != 0 and len(verify['body_vectors']) != 0:
                score[1] = mean_strategy_cal_scores(v1_body, verify['body_vectors'])
            if sum(score.values()) == 0:
                continue
            score = score[1] if len(v1) == 0 or (score[0] == 0 and score[1] > [0.9069641, 0.985643][type_ - 1]) else score[0]
            # score = score[1] if len(v1) == 0 or (score[0] == 0 and score[1] > [0.9069641, 0.987452][type_ - 1]) else score[0]
            l.append((f2, score))
        l = sorted(l, key=lambda x: x[1], reverse=True)
        answer = [l[i][0] for i in range(min(100, len(l)))]
        if l:
            df.append(
                (
                    str(f.name),
                    l[0][1],
                    np.mean([l[i][1] for i in range(3)]),
                    np.mean([l[i][1] for i in range(10)]),
                    ','.join([str(i.name) for i in answer])
                )
            )

    return df


def create_table(db: Dict[Path, Any]) -> pd.DataFrame:
    columns = (
        'query',
        'matched_1',
        'matched_3',
        'matched_10',
        'answer'
    )
    l = []
    for big_folder in db:
        df = calc_scores(*db[big_folder])
        l.extend(df)

    df = pd.DataFrame(data=l, columns=columns)
    return df


@torch.no_grad()
def main():
    device = 'cuda:0'

    body_preproc = Preproc4(device=device)

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
    ).eval().model_loss

    cat_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/10/cfc4f63a71c2417087adb2d3feb5d34c/artifacts/checkpoints'
                '/10/cfc4f63a71c2417087adb2d3feb5d34c/checkpoints/epoch=42-step=51943.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/10/cfc4f63a71c2417087adb2d3feb5d34c/artifacts/cat_fe_head.py')
    ).eval().model_loss
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

    dog_body_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/1/f6fca5573f62410ab0649407c951c153/artifacts/checkpoints'
                '/1/f6fca5573f62410ab0649407c951c153/checkpoints/epoch=36-step=42142.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/11/84ff365352fa4e058b30c882bc00a607/artifacts/body_dog_fe.py')
    ).eval().model_loss

    cat_body_model = Controller.load_from_checkpoint(
        str(
            Path(
                'mlruns/11/84ff365352fa4e058b30c882bc00a607/artifacts/checkpoints'
                '/11/84ff365352fa4e058b30c882bc00a607/checkpoints/epoch=39-step=100559.ckpt'
            )
        ),
        config=get_dict_wrapper('mlruns/13/6502b10363974f0f825e709b522ee659/artifacts/body_cat_fe.py')
    ).eval().model_loss
    # cat_body_model = Controller.load_from_checkpoint(
    #     str(
    #         Path(
    #             'mlruns/13/7c1fc7d0fae74936b9297c6669335aa0/artifacts/checkpoints'
    #             '/13/7c1fc7d0fae74936b9297c6669335aa0/checkpoints/epoch=6-step=14244.ckpt'
    #         )
    #     ),
    #     config=get_dict_wrapper('mlruns/13/7c1fc7d0fae74936b9297c6669335aa0/artifacts/body_cat_fe.py')
    # ).eval().model_loss
    dog_body_model.add_margin = None
    cat_body_model.add_margin = None
    dog_body_model.to(device)
    cat_body_model.to(device)

    models = {
        1: dog_model,
        2: cat_model
    }
    body_models = {
        1: dog_body_model,
        2: cat_body_model
    }

    def pipeline(img, type_):
        with suppress(AssertionError, ValueError, OSError, cv2.error):
            head_img = head_preproc(img)
            head_img_tensor = torch.tensor(head_img).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255

            vector = models[type_](head_img_tensor).cpu()
            return vector
        return None

    def body_pipeline(img, type_):
        try:
            body_img = body_preproc(img)
        except (AssertionError, ValueError, OSError, cv2.error):
            return None
        body_img = np.array(resize_with_padding(Image.fromarray(body_img)))
        body_img_tensor = torch.tensor(body_img).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255

        vector = body_models[type_](body_img_tensor).cpu()
        return vector

    db_path = Path('scores3.pickle')
    if db_path.exists():
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
    else:
        db = prepare_data(Path('../pets_datasets/_blip_split_v3_public/test'), pipeline, body_pipeline)
        with open(db_path, 'wb') as f:
            pickle.dump(db, f)

    df = create_table(db)

    df.to_csv('pred_scores_test4.tsv', index=False, sep='\t')


if __name__ == '__main__':
    main()
    df1 = pd.read_csv('pred_scores_test4.tsv', sep='\t')
    df2 = pd.read_csv('..\\kashtanka_pet_scoring-master\\preds.tsv', sep='\t')
    d1 = {row['query']: row for _, row in df1.iterrows()}
    d2 = {row['query']: row for _, row in df2.iterrows()}
    d = []
    for q, row in d2.items():
        if q in d1:
            d.append(d1[q])
        else:
            d.append(d2[q])
    df_final = pd.DataFrame(d, columns=df1.columns)
    df_final.to_csv('..\\kashtanka_pet_scoring-master\\pred_scores_test4.tsv', index=False, sep='\t')
