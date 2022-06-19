import json
import pickle
from ast import literal_eval
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score


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


def void(*_, **__):
    ...


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


def evaluate(preds, scores, g_t):
    metrics = {}
    ious = []
    preds_copy = preds
    scores_copy = scores
    g_t_copy = g_t
    for thr in (0.5, 0.7, 0.75, 0.9):
        results = []
        preds = deepcopy(preds_copy)
        scores = deepcopy(scores_copy)
        g_t = deepcopy(g_t_copy)
        for j in range(len(preds)):
            for a in range(len(preds[j])):
                dt = preds[j][a]
                results.append({'score': scores[j][a]})
                ious = [intersection_over_union(g_t[j][b], dt) for b in range(len(g_t[j]))]
                if len(ious) == 0:
                    max_IoU = -1
                    max_gt_id = -1
                else:
                    max_gt_id, max_IoU = max([i for i in zip(list(range(len(ious))), ious)], key=lambda x: x[1])
                if max_gt_id >= 0 and max_IoU >= thr:
                    results[-1]['TP'] = 1
                    g_t[j] = np.delete(g_t[j], max_gt_id, axis=0)
                    if thr == 0.5:
                        ious.append(max_IoU)
                else:
                    if thr == 0.5:
                        ious.append(0)
                    results[-1]['TP'] = 0

        results = sorted(results, key=lambda k: k['score'], reverse=True)
        score = [i['score'] for i in results]
        flags = [i['TP'] for i in results]

        if len(flags) == 0:
            AP = 0.0
        else:
            AP = average_precision_score(flags, score)
        metrics[f'AP at {thr}'] = AP
    metrics['IoU'] = np.mean(ious)
    return metrics


def compute_scores_tsinghua(df, mode):
    void(df, mode)
    raise NotImplementedError


def compute_scores_oxford(df, mode):
    void(df, mode)
    raise NotImplementedError


def compute_scores_data_25(df, mode):
    with open('data_25_anno.pickle', 'rb') as f:
        db = pickle.load(f)

    cut_db = [{}, {}]
    for i in range(len(db)):
        for k, v in db[i].items():
            detections = []
            with suppress(KeyError):
                for j in range(len(v)):
                    t = v[j][mode]
                    h, w = v[j]['resolution']
                    t = [t['x'], t['y'], t['x'] + t['width'], t['y'] + t['height']]
                    t = [t[0] * w / 100, t[1] * h / 100, t[2] * w / 100, t[3] * h / 100]
                    t = np.round(t).astype(int)
                    detections.append(t.tolist())
            if detections:
                cut_db[i][k] = detections

    d = {row['query']: row for _, row in df.iterrows()}

    for tag, i in zip(('Dog', 'Cat'), range(len(cut_db))):
        preds = []
        g_t = []
        scores = []
        for k, true_detections in cut_db[i].items():
            g_t.append(true_detections)
            if k in d:
                preds.append(literal_eval(d[k]['detections']))
                scores.append(literal_eval(d[k]['scores']))
            else:
                preds.append([])
                scores.append([])
        metrics = evaluate(preds, scores, g_t)
        print(*[f'{tag} {mode} {k} = {v}' for k, v in metrics.items()], sep='\n')
    print()


available_ds = {
    'data_25': compute_scores_data_25,
    'oxford': compute_scores_oxford,
    'tsinghua': compute_scores_tsinghua
}


def main(path: str, ds: str, mode: str):
    path = Path(path)
    assert path.exists(), 'Incorrect path to the .tsv file'
    assert ds in available_ds.keys(), f'Invalid ds has been entered. Please choose from {tuple(available_ds)}'
    assert mode in ('Head', 'Animal'), 'Invalid mode. Please choose from {}'.format(('Head', 'Animal'))
    df = pd.read_csv(path, sep='\t')
    assert all(i in df.columns for i in ('query', 'detections', 'scores')), 'Incorrectly formatted .tsv file'
    available_ds[ds](df, mode)


if __name__ == '__main__':
    # parse_labeled_studio(Path('../pets_datasets/sampled results'), Path('../pets_datasets/data_25_labeled'))
    # main('detected_body.tsv', 'data_25', 'Animal')
    # main('detected_head.tsv', 'data_25', 'Head')
    fire.Fire(main)
