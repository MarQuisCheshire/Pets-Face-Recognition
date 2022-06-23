import json
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
            score = score[1] if len(v1) == 0 or (score[0] == 0 and score[1] > [0.9069641, 0.985643][type_ - 1]) else \
            score[0]
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

    dog_model = Controller(get_dict_wrapper('configs/to_reproduce/dog_fe/fe_dogs_config.py'))
    dog_model.load_state_dict(
        torch.load(str(Path('configs/to_reproduce/dog_fe/epoch=36_head.ckpt'))),
        strict=False
    )
    dog_model = dog_model.eval().model_loss

    cat_model = Controller(get_dict_wrapper('configs/to_reproduce/cat_fe/cat_fe_head.py'))
    cat_model.load_state_dict(
        torch.load(str(Path('configs/to_reproduce/cat_fe/epoch=42_head.ckpt'))),
        strict=False
    )
    cat_model = cat_model.eval().model_loss
    dog_model.add_margin = None
    cat_model.add_margin = None
    dog_model.to(device)
    cat_model.to(device)

    dog_body_model = Controller(get_dict_wrapper('configs/to_reproduce/dog_fe/body_dog_fe.py'))
    dog_body_model.load_state_dict(
        torch.load(str(Path('configs/to_reproduce/dog_fe/epoch=37_body.ckpt'))),
        strict=False
    )
    dog_body_model = dog_body_model.eval().model_loss

    cat_body_model = Controller(get_dict_wrapper('configs/to_reproduce/cat_fe/body_cat_fe.py'))
    cat_body_model.load_state_dict(
        torch.load(str(Path('configs/to_reproduce/cat_fe/epoch=39_body.ckpt'))),
        strict=False
    )
    cat_body_model = cat_body_model.eval().model_loss

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

    db = prepare_data(Path('../pets_datasets/_blip_split_v3_public/test'), pipeline, body_pipeline)

    df = create_table(db)

    df.to_csv('pred_scores_test1.tsv', index=False, sep='\t')


if __name__ == '__main__':
    main()

    # add random rows that have no predictions
    df1 = pd.read_csv('pred_scores_test1.tsv', sep='\t')
    df2 = pd.read_csv('preds.tsv', sep='\t')
    d1 = {row['query']: row for _, row in df1.iterrows()}
    d2 = {row['query']: row for _, row in df2.iterrows()}
    d = []
    for q, row in d2.items():
        if q in d1:
            d.append(d1[q])
        else:
            d.append(d2[q])
    df_final = pd.DataFrame(d, columns=df1.columns)
    df_final.to_csv('pred_scores_test1.tsv', index=False, sep='\t')
