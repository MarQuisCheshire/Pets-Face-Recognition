import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Optional, List, Callable, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pipe import chain, where, select
from tqdm import tqdm

from data_loading import RecDataset
from preprocessor import Preproc4, Preproc3, Preproc6, Preproc7, Preproc8, Preproc10, Preproc9, Preproc12, Preproc11

v = ''


def transform_dataset(
        input_root: Union[Path, str],
        preprocessor: Callable,
        output_root: Union[Path, str, None] = None,
        paths: Optional[List[Union[Path, str]]] = None,
        out_paths: Optional[List[Union[Path, str]]] = None
) -> None:
    input_root = Path(input_root)
    if paths is None:
        paths = (input_root.glob('*/*.jpg'), input_root.glob('*/*.png')) | chain
    else:
        str_input_path = str(input_root.resolve())
        assert all(str_input_path in str(Path(path).resolve()) for path in paths), \
            'All paths should be children of the input root'

    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    paths = list(paths)
    for i in tqdm(range(len(paths))):
        with suppress(AssertionError, ValueError, OSError, cv2.error):
            if out_paths is None:
                rel_path = os.path.relpath(paths[i], input_root)
                rel_path = output_root / rel_path
            else:
                rel_path = Path(out_paths[i])
            if not rel_path.exists() and not (rel_path.parent / (rel_path.name[:-4] + '.jpg')).exists():
                image = np.array(Image.open(paths[i]).convert('RGB'))
                processed_image = preprocessor(image)
                rel_path.parent.mkdir(exist_ok=True)
                if processed_image.shape[0] * processed_image.shape[1] > 300 * 400:
                    rel_path = rel_path.parent / (rel_path.name[:-4] + '.jpg')
                Image.fromarray(processed_image).save(rel_path)


def data_25(preprocessor, type_=1):
    assert type_ in (1, 2)
    paths_to_exclude = [
        'data_25\\rl131336\\216319.jpg',
        'data_25\\rl378360\\660074.jpg',
        'data_25\\rf337006\\589105.jpg',
        'data_25\\rl341945\\597666.jpg',
        'data_25\\rl254355\\447992.jpg',
        'data_25\\rl302213\\529924.jpg',
        'data_25\\rf327026\\572016.jpg',
        'data_25\\rf287909\\505121.jpg',
        'data_25\\rf413612\\717733.jpg',
        'data_25\\rl257226\\452879.jpg',
        'data_25\\rl257226\\452880.jpg',
        'data_25\\rl411182\\713855.jpg',
        'data_25\\rf292282\\512681.jpg',
        'data_25\\rf263807\\464166.jpg',
        'data_25\\rf146140\\246925.jpg',
        'data_25\\rf230595\\407467.jpg',
        'data_25\\rl209386\\373061.jpg',
        'data_25\\rf428033\\742644.jpg',
        'data_25\\rl270079\\474803.jpg',
        'data_25\\rf278099\\488547.jpg',
        'data_25\\rl401247\\697651.jpg',
        'data_25\\rl381795\\666073.jpg',
        'data_25\\rf233445\\412363.jpg',
        'data_25\\rl223935\\650763.jpg',
        'data_25\\rl343571\\600399.jpg',
        'data_25\\rf337006\\589105.jpg',
        'data_25\\rl381795\\666046.jpg',
        'data_25\\rl381795\\666053.jpg',
        'data_25\\rl381795\\666059.jpg',
        'data_25\\rl381795\\666067.jpg',
        'data_25\\rl381795\\666073.jpg',
        'data_25\\rl381795\\666077.jpg',
        'data_25\\rl381795\\666081.jpg',
        'data_25\\rl381795\\666089.jpg',
        'data_25\\rl381795\\666094.jpg',
        'data_25\\rl381795\\666097.jpg',
        'data_25\\rl381795\\666103.jpg',

        'data_25\\rf133909\\221703.jpg',
        'data_25\\rf133909\\221704.jpg',
        'data_25\\rf133909\\221705.jpg',

        'data_25\\rf133831\\221554.jpg',
        'data_25\\rf133831\\221555.jpg',
        'data_25\\rf133831\\221556.jpg',
    ]
    paths_to_exclude = [(Path('../pets_datasets') / i).resolve() for i in paths_to_exclude]
    cats = RecDataset(Path('../pets_datasets/data_25'), type_, 1, paths_to_exclude=paths_to_exclude)
    paths = [cats.index_to_path[i] for i in range(len(cats))]
    transform_dataset(
        Path('../pets_datasets/data_25'),
        preprocessor,
        Path(f'../pets_datasets/data_25_transformed_{v}_{"dog" if type_ == 1 else "cat"}s'),
        paths
    )


def petfinder(preprocessor):
    p1 = Path('../pets_datasets/petfinder/train_metadata')
    p2 = Path('../pets_datasets/petfinder/train_images')

    json_paths = {i.name[:-5]: i for i in p1.glob('*.json')}

    def check_type(p: Path) -> bool:
        json_p = json_paths[p.name[:-4]]
        with open(json_p, 'r', encoding='utf8') as f:
            info = json.load(f)
        with suppress(KeyError):
            return info['labelAnnotations'][0]['description'] == 'dog'
        return False

    paths = list((p2.glob('*.jpg'), p2.glob('*.png')) | chain | where(check_type) | select(lambda x: x.resolve()))
    outp = Path(f'../pets_datasets/petfinder_transformed_{v}_dogs')
    outp.mkdir(parents=True)
    out_paths = [outp / i.name.split('-')[0] / i.name for i in paths]

    transform_dataset(
        p2,
        preprocessor,
        paths=paths,
        out_paths=out_paths
    )


def extra_petfinder(preprocessor, tag='dog'):
    if tag == 'dog':
        p1 = Path(f'../pets_datasets/petfinder_extra_dogs_transformed_{v}')
        p2 = Path('../pets_datasets/petfinder_extra_dogs')

        paths_to_exclude = (
                [i for i in (p2 / '48683845').iterdir()] +
                [i for i in (p2 / '45528036').iterdir()] +
                [p2 / '48009947' / '3.png']
        )
        paths_to_exclude = {i.resolve() for i in paths_to_exclude}
    elif tag == 'cat':
        p1 = Path(f'../pets_datasets/petfinder_extra_cats_transformed_{v}')
        p2 = Path('../pets_datasets/petfinder_extra_cats')

        paths_to_exclude = (
            [p2 / '24355557' / '4.png']
        )
        paths_to_exclude = {i.resolve() for i in paths_to_exclude}

    paths = [j.resolve() for i in p2.resolve().iterdir() for j in i.iterdir() if j.resolve() not in paths_to_exclude]

    transform_dataset(
        p2,
        preprocessor,
        output_root=p1,
        paths=paths,
    )


def labeled_data(preprocessor):
    preprocessor.return_for_metrics = True
    p = Path('../pets_datasets/data_25_labeled').resolve()
    data = []
    if isinstance(preprocessor, (Preproc4, Preproc6, Preproc8, Preproc10, Preproc12)):
        for input_root in p.iterdir():

            input_root = Path(input_root)

            paths = (input_root.glob('*/*.jpg'), input_root.glob('*/*.png')) | chain

            paths = list(paths)
            for i in tqdm(range(len(paths))):
                with suppress(AssertionError, ValueError, OSError, cv2.error):
                    image = np.array(Image.open(paths[i]).convert('RGB'))
                    bbox, score = preprocessor(image)
                    bbox = [bbox.tolist()]
                    score = list(score)
                    data.append((paths[i].name, bbox, score))
        df = pd.DataFrame(data, columns=('query', 'detections', 'scores'))
        if isinstance(preprocessor, Preproc4):
            df.to_csv('detected_body.tsv', index=False, sep='\t')
        else:
            df.to_csv('detected_head4.tsv', index=False, sep='\t')
    elif isinstance(preprocessor, (Preproc3, Preproc7, Preproc9, Preproc11)):
        for input_root in p.iterdir():

            input_root = Path(input_root)

            paths = (input_root.glob('*/*.jpg'), input_root.glob('*/*.png')) | chain

            paths = list(paths)
            for i in tqdm(range(len(paths))):
                with suppress(AssertionError, ValueError, OSError, cv2.error):
                    image = np.array(Image.open(paths[i]).convert('RGB'))
                    pts = preprocessor(image).tolist()
                    data.append((paths[i].name, *pts))
        df = pd.DataFrame(data, columns=('query', 'Left eye', 'Right eye', 'Nose'))
        df.to_csv('landmark4.tsv', index=False, sep='\t')
    else:
        raise Exception


if __name__ == '__main__':
    # preproc cat+dog v3
    # preprocessor = Preproc3(
    #     np.array([[43, 53], [83, 53], [63, 83]]),
    #     (128, 128, 3),
    #     (0, 0, 0),
    #     device='cuda:0',
    #     old_align=True
    # )
    # preproc v5
    # preprocessor = Preproc3(
    #     np.array([[64, 92], [160, 92], [112, 180]]),
    #     (224, 224, 3),
    #     (0, 0, 0),
    #     device='cuda:0'
    # )
    # extra_petfinder(preprocessor)
    # v6
    # preprocessor = Preproc3(
    #     np.array([[70, 92], [154, 92], [112, 160]]),
    #     (224, 224, 3),
    #     (0, 0, 0),
    #     device='cuda:0'
    # )
    # v = 'v6'
    # preproc dog v4
    # preprocessor = Preproc4(device='cuda:0')
    # v = 'v4'
    # extra_petfinder(preprocessor)

    # preprocessor = Preproc4(device='cuda:0', masked=True, mask_thr=0.7)
    # v = 'v4_masked'
    # extra_petfinder(preprocessor)
    # data_25(preprocessor, 1)
    # data_25(preprocessor, 2)
    # extra_petfinder(preprocessor, 'cat')

    # dump data
    # preprocessor = Preproc6(device='cuda:0')
    # preprocessor = Preproc8(device='cuda:0')
    preprocessor = Preproc11(
        np.array([[70, 92], [154, 92], [112, 160]]),
        (224, 224, 3),
        (0, 0, 0),
        device='cuda:0'
    )
    labeled_data(preprocessor)
    preprocessor = Preproc12(device='cuda:0')
    labeled_data(preprocessor)
