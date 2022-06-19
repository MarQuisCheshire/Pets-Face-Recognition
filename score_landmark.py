import json
import pickle
from ast import literal_eval
from contextlib import suppress
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from PIL import Image


def parse_labeled_studio(p, p2):
    processed = [{}, {}]

    img_d_p = {j.name: j for i in p2.resolve().iterdir() for k in i.iterdir() for j in k.iterdir()}

    for case in ('old', 'new'):
        for ids in (p / case).iterdir():
            for js in ids.iterdir():
                with open(js, 'r') as f:
                    t = json.load(f)
                animal_type = ['dog', 'cat'].index(js.name[:-5])
                for i in range(len(t)):
                    tmp = []
                    img_name = '-'.join(t[i]['file_upload'].split('-')[1:])
                    for k in range(len(t[i]['annotations'])):
                        tmp.append({})
                        for j in t[i]['annotations'][k]['result']:
                            if 'keypointlabels' in j['value']:
                                tmp[-1][j['value']['keypointlabels'][0]] = j['value']
                            else:
                                tmp[-1][j['value']['rectanglelabels'][0]] = j['value']
                        tmp[-1]['resolution'] = np.array(Image.open(img_d_p[img_name]).convert('RGB')).shape[:-1]
                    processed[animal_type][img_name] = tmp

    with open('data_25_anno.pickle', 'wb') as f:
        pickle.dump(processed, f)


def evaluate(preds, g_t, l):
    metrics = {}

    to_average = []
    ds = []
    for i in range(len(g_t)):
        d = ((g_t[i][0] - g_t[i][1]) ** 2).sum() ** 0.5
        ds.append(d)
        nme = ((preds[i][:-1] - g_t[i][:-1]) ** 2).sum(axis=1) ** 0.5 / d[None]
        to_average.extend(nme)

    to_average = np.asarray(to_average)
    metrics['Length'] = len(to_average)
    metrics['NME'] = np.mean(to_average)
    metrics['NME 0.05 0.95'] = to_average[(to_average > np.quantile(to_average, 0.05)) & (to_average < np.quantile(to_average, 0.95))].mean()
    metrics['NME median'] = np.median(to_average)
    metrics['NME 0.75'] = np.quantile(to_average, 0.75)
    metrics['NME 0.25'] = np.quantile(to_average, 0.25)

    return metrics


def compute_scores_data_25(df):
    with open('data_25_anno.pickle', 'rb') as f:
        db = pickle.load(f)

    cut_db = [{}, {}]
    for i in range(len(db)):
        for k, v in db[i].items():
            detections = []
            with suppress(KeyError):
                for j in range(len(v)):
                    a = []
                    for mode in ('Left eye', 'Right eye', 'Nose'):
                        t = v[j][mode]
                        t = [t['x'], t['y']]
                        t = np.round(t).astype(int)
                        a.append(t)
                    h, w = v[j]['resolution']
                    detections.append(np.array(a) * np.asarray([w, h])[None] / 100)
            if detections:
                cut_db[i][k] = detections[0]

    d = {row['query']: row for _, row in df.iterrows()}

    for tag, i in zip(('Dog', 'Cat'), range(len(cut_db))):
        preds = []
        g_t = []
        l = []
        for k, true_detections in cut_db[i].items():
            with suppress(KeyError):
                preds.append(np.array((
                    literal_eval(d[k]['Left eye']),
                    literal_eval(d[k]['Right eye']),
                    literal_eval(d[k]['Nose'])
                )))
                g_t.append(true_detections)

                l.append(k)

        metrics = evaluate(preds, g_t, l)
        print(*[f'{tag} {k} = {v}' for k, v in metrics.items()], sep='\n')
    print()


available_ds = {
    'data_25': compute_scores_data_25,
}


def main(path: str, ds: str):
    path = Path(path)
    assert path.exists(), 'Incorrect path to the .tsv file'
    assert ds in available_ds.keys(), f'Invalid ds has been entered. Please choose from {tuple(available_ds)}'
    df = pd.read_csv(path, sep='\t')
    assert all(i in df.columns for i in ('query', 'Left eye', 'Right eye', 'Nose')), 'Incorrectly formatted .tsv file'
    available_ds[ds](df)


if __name__ == '__main__':
    # main('landmark.tsv', 'data_25')
    fire.Fire(main)
