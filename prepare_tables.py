from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pipe import chain
from tqdm import tqdm

from preprocessor import Preproc4, Preproc3, Preproc6, Preproc7, Preproc8, Preproc10, Preproc9, Preproc12, Preproc11


def preapre_table(preprocessor):
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
            df.to_csv('detected_head.tsv', index=False, sep='\t')
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
        df.to_csv('landmark.tsv', index=False, sep='\t')
    else:
        raise Exception


if __name__ == '__main__':
    preprocessor = Preproc3(
        np.array([[70, 92], [154, 92], [112, 160]]),
        (224, 224, 3),
        (0, 0, 0),
        device='cuda:0'
    )
    preapre_table(preprocessor)

    preprocessor = Preproc4(device='cuda:0', masked=True, mask_thr=0.7)
    preapre_table(preprocessor)

    preprocessor = Preproc6(device='cuda:0')
    preapre_table(preprocessor)
