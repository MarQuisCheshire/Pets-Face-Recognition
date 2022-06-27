from pathlib import Path

import numpy as np
import torch
import torchvision
from imutils import face_utils

from engine import DetectionController, KeyPointsController
from utils import get_dict_wrapper
from .align import align

try:
    import dlib

    dlib_flag = True
except ModuleNotFoundError:
    dlib_flag = False

if dlib_flag:

    class DogPreproc:

        def __init__(self, base_pts, dsize, padding_val):
            self.base_pts = base_pts
            self.dsize = dsize
            self.padding_val = padding_val
            self.detector = None
            self.predictor = None
            # self.to_tensor = None
            self.models_init()

        def __call__(self, img):
            dets = self.detector(img, upsample_num_times=1)

            d = None
            s = 0
            for i in dets:
                x1, y1 = i.rect.left(), i.rect.top()
                x2, y2 = i.rect.right(), i.rect.bottom()
                if s < abs(x2 - x1) * abs(y2 - y1):
                    s = abs(x2 - x1) * abs(y2 - y1)
                    d = i

            shape = face_utils.shape_to_np(self.predictor(img, d.rect))
            if len(self.base_pts) == 3:
                pts = [shape[5], shape[2], shape[3]]
            else:
                pts = [shape[4], shape[1], shape[5], shape[2]]
            aimg = align(img, pts, self.base_pts, self.dsize, self.padding_val)
            # aimg = self.to_tensor(aimg)
            return aimg

        def models_init(self):
            self.detector = dlib.cnn_face_detection_model_v1('preprocessor/dog_face_detector/dogHeadDetector.dat')
            self.predictor = dlib.shape_predictor('preprocessor/dog_face_detector/landmarkDetector.dat')
            # self.to_tensor = torchvision.transforms.ToTensor()

        def __getstate__(self):
            d = {k: v for k, v in self.__dict__.items() if k not in ['detector', 'predictor', 'to_tensor']}
            return d

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.models_init()


    class CatPreproc:

        def __init__(self, base_pts, dsize, padding_val):
            self.base_pts = base_pts
            self.dsize = dsize
            self.padding_val = padding_val
            self.detector = None
            self.predictor = None
            self.to_tensor = None
            self.models_init()

        def __call__(self, img):
            dets = self.detector(img, upsample_num_times=1)

            d = None
            s = 0
            for i in dets:
                x1, y1 = i.rect.left(), i.rect.top()
                x2, y2 = i.rect.right(), i.rect.bottom()
                if s < abs(x2 - x1) * abs(y2 - y1):
                    s = abs(x2 - x1) * abs(y2 - y1)
                    d = i

            shape = face_utils.shape_to_np(self.predictor(img, d.rect))
            if len(self.base_pts) == 3:
                pts = [shape[5], shape[2], shape[3]]
            else:
                pts = [shape[4], shape[1], shape[5], shape[2]]
            aimg = align(img, pts, self.base_pts, self.dsize, self.padding_val)
            aimg = self.to_tensor(aimg)
            return aimg

        def models_init(self):
            self.detector = dlib.fhog_object_detector('preprocessor/pycatfd-master/data/detector.svm')
            self.predictor = dlib.shape_predictor('preprocessor/pycatfd-master/data/predictor.dat')
            self.to_tensor = torchvision.transforms.ToTensor()

        def __getstate__(self):
            d = {k: v for k, v in self.__dict__.items() if k not in ['detector', 'predictor', 'to_tensor']}
            return d

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.models_init()


class DogPreproc2:

    def __init__(self, base_pts, dsize, padding_val):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.predictor = None
        # self.to_tensor = None
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255)
        dets = dets[0]['boxes'].cpu().detach().numpy()

        d = dlib.rectangle(*dets[0].astype(int))

        shape = face_utils.shape_to_np(self.predictor(img, d))
        if len(self.base_pts) == 3:
            pts = [shape[5], shape[2], shape[3]]
        else:
            pts = [shape[4], shape[1], shape[5], shape[2]]
        aimg = align(img, pts, self.base_pts, self.dsize)
        # aimg = self.to_tensor(aimg)
        return aimg

    def models_init(self):
        self.detector = DetectionController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/4/41b8298f569342e9bdd1b0362b03478e/artifacts/checkpoints/'
                    '4/41b8298f569342e9bdd1b0362b03478e/checkpoints/epoch=58-step=10796.ckpt'
                )
            ),
            config=get_dict_wrapper('mlruns/4/41b8298f569342e9bdd1b0362b03478e/artifacts/detection_config1.py')
        ).eval().model_loss
        self.predictor = dlib.shape_predictor('preprocessor/dog_face_detector/landmarkDetector.dat')

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector', 'predictor', 'to_tensor']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc3:

    def __init__(self, base_pts, dsize, padding_val, thr=0.9, min_distance=5, device='cpu', old_align=False):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.thr = thr
        self.min_distance = min_distance
        self.device = device
        self.old_align = old_align
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        pts = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        pts = pts[0]['keypoints'].cpu().detach().numpy()
        pts = np.round(pts[0, :, :-1]).astype(int)

        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(np.sqrt(((pts[i] - pts[j]) ** 2).sum()))
        assert all(i > self.min_distance for i in d)

        if self.return_for_metrics:
            return pts

        aimg = align(img, pts, self.base_pts, self.dsize)
        return aimg

    def models_init(self):
        # self.detector = KeyPointsController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/checkpoints/'
        #             '9/4100f0feaa39434b92b56b5faf000d97/checkpoints/epoch=14-step=62384.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/keypoints_config.py')
        # ).eval().model_loss
        self.detector = KeyPointsController(get_dict_wrapper(Path('configs/to_reproduce/keypoint/keypoints_config.py')))
        self.detector.load_state_dict(torch.load(Path('configs/to_reproduce/keypoint/epoch=14.ckpt')))
        self.detector = self.detector.eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc4:

    def __init__(self, thr=0.9, mask_thr=0.5, device='cpu', masked=False):
        self.detector = None
        self.thr = thr
        self.mask_thr = mask_thr
        self.device = device
        self.masked = masked
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)
        mask = (dets[0]['masks'][0, 0].cpu().detach().numpy() > self.mask_thr).astype(int)

        if self.masked:
            aimg = img * mask[:, :, None]
            bbox[0] = max(bbox[0], (mask.sum(axis=0) == 0).tolist().index(False))
            bbox[1] = max(bbox[1], (mask.sum(axis=1) == 0).tolist().index(False))
            bbox[2] = min(bbox[2], mask.shape[1] - (mask.sum(axis=0) == 0).tolist()[::-1].index(False))
            bbox[3] = min(bbox[3], mask.shape[0] - (mask.sum(axis=1) == 0).tolist()[::-1].index(False))
        else:
            aimg = img

        if self.return_for_metrics:
            return bbox, score
        aimg = aimg[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg.astype(np.uint8)

    def models_init(self):
        # self.detector = DetectionController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/checkpoints/'
        #             '8/e0659336f7d3449fb7b2eac767df655f/checkpoints/epoch=64-step=16249.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/mask_rcnn_config.py')
        # ).eval().model_loss

        self.detector = DetectionController(get_dict_wrapper(Path('configs/to_reproduce/mask/mask_rcnn_config.py')))
        self.detector.load_state_dict(torch.load(Path('configs/to_reproduce/mask/epoch=64.ckpt')))
        self.detector = self.detector.eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc6:

    def __init__(self, thr=0.9, device='cpu'):
        self.detector = None
        self.thr = thr
        self.device = device
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)

        if self.return_for_metrics:
            return bbox, score
        aimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg

    def models_init(self):
        # self.detector = KeyPointsController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/checkpoints/'
        #             '9/4100f0feaa39434b92b56b5faf000d97/checkpoints/epoch=14-step=62384.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/keypoints_config.py')
        # ).eval().model_loss
        self.detector = KeyPointsController(get_dict_wrapper(Path('configs/to_reproduce/keypoint/keypoints_config.py')))
        self.detector.load_state_dict(torch.load(Path('configs/to_reproduce/keypoint/epoch=14.ckpt')))
        self.detector = self.detector.eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc8:

    def __init__(self, thr=0.9, device='cpu'):
        self.detector = None
        self.thr = thr
        self.device = device
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)

        if self.return_for_metrics:
            return bbox, score
        aimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/654e1dcc13534afaae6c2069bae2d55b/artifacts/checkpoints'
                    '/9/654e1dcc13534afaae6c2069bae2d55b/checkpoints/epoch=11-step=9611.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/654e1dcc13534afaae6c2069bae2d55b/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc7:

    def __init__(self, base_pts, dsize, padding_val, thr=0.7, min_distance=5, device='cpu', old_align=False):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.thr = thr
        self.min_distance = min_distance
        self.device = device
        self.old_align = old_align
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        pts = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        pts = pts[0]['keypoints'].cpu().detach().numpy()
        pts = np.round(pts[0, :, :-1]).astype(int)

        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(np.sqrt(((pts[i] - pts[j]) ** 2).sum()))
        assert all(i > self.min_distance for i in d)

        if self.return_for_metrics:
            return pts

        aimg = align(img, pts, self.base_pts, self.dsize)
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/654e1dcc13534afaae6c2069bae2d55b/artifacts/checkpoints'
                    '/9/654e1dcc13534afaae6c2069bae2d55b/checkpoints/epoch=11-step=9611.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/654e1dcc13534afaae6c2069bae2d55b/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc10:

    def __init__(self, thr=0.9, device='cpu'):
        self.detector = None
        self.thr = thr
        self.device = device
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)

        if self.return_for_metrics:
            return bbox, score
        aimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/artifacts/checkpoints'
                    '/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/checkpoints/epoch=5-step=5411.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc9:

    def __init__(self, base_pts, dsize, padding_val, thr=0.9, min_distance=5, device='cpu', old_align=False):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.thr = thr
        self.min_distance = 5
        self.device = device
        self.old_align = old_align
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        pts = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        pts = pts[0]['keypoints'].cpu().detach().numpy()
        pts = np.round(pts[0, :, :-1]).astype(int)

        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(np.sqrt(((pts[i] - pts[j]) ** 2).sum()))
        assert all(i > self.min_distance for i in d)

        if self.return_for_metrics:
            return pts

        aimg = align(img, pts, self.base_pts, self.dsize)
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/artifacts/checkpoints'
                    '/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/checkpoints/epoch=5-step=5411.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/8b15c649e3df4ee9a7a4cbed6be1b1c9/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc12:

    def __init__(self, thr=0.9, device='cpu'):
        self.detector = None
        self.thr = thr
        self.device = device
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)

        if self.return_for_metrics:
            return bbox, score
        aimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/8d5b5fd125c6414389721c6e77ae5306/artifacts/checkpoints'
                    '/9/8d5b5fd125c6414389721c6e77ae5306/checkpoints/epoch=12-step=51973.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/8d5b5fd125c6414389721c6e77ae5306/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc11:

    def __init__(self, base_pts, dsize, padding_val, thr=0.9, min_distance=5, device='cpu', old_align=False):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.thr = thr
        self.min_distance = 5
        self.device = device
        self.old_align = old_align
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        pts = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        pts = pts[0]['keypoints'].cpu().detach().numpy()
        pts = np.round(pts[0, :, :-1]).astype(int)

        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(np.sqrt(((pts[i] - pts[j]) ** 2).sum()))
        assert all(i > self.min_distance for i in d)

        if self.return_for_metrics:
            return pts

        aimg = align(img, pts, self.base_pts, self.dsize)
        return aimg

    def models_init(self):
        self.detector = KeyPointsController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/9/8d5b5fd125c6414389721c6e77ae5306/artifacts/checkpoints'
                    '/9/8d5b5fd125c6414389721c6e77ae5306/checkpoints/epoch=12-step=51973.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/9/8d5b5fd125c6414389721c6e77ae5306/artifacts/keypoints_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class Preproc5:

    def __init__(self, thr=0.9, mask_thr=0.5, device='cpu'):
        self.detector = None
        self.thr = thr
        self.mask_thr = mask_thr
        self.device = device
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        dets = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = np.round(dets[0]['boxes'][0].cpu().detach().numpy()).astype(int)
        mask = dets[0]['masks'][0, 0].cpu().detach().numpy()
        mask[mask < self.mask_thr] = mask[mask < self.mask_thr] ** 2
        mask[mask >= self.mask_thr] = 1

        aimg = img * mask[:, :, None]
        aimg = aimg[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        return aimg.astype(np.uint8)

    def models_init(self):
        self.detector = DetectionController.load_from_checkpoint(
            str(
                Path(
                    'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/checkpoints/'
                    '8/e0659336f7d3449fb7b2eac767df655f/checkpoints/epoch=64-step=16249.ckpt'
                )
            ),
            config=get_dict_wrapper(r'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/mask_rcnn_config.py')
        ).eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class PreprocCombined:

    def __init__(self, base_pts, dsize, padding_val, thr=0.9, mask_thr=0.7, device='cpu', min_distance=5,):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.thr = thr
        self.min_distance = min_distance
        self.detector = None
        self.lmd = None
        self.thr = thr
        self.mask_thr = mask_thr
        self.device = device
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        tensor_img = torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255
        pts = self.lmd(tensor_img)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        dets = self.detector(tensor_img)
        score = dets[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        mask = (dets[0]['masks'][0, 0].cpu().detach().numpy() > self.mask_thr).astype(int)

        aimg = (img * mask[:, :, None]).astype(np.uint8)

        pts = pts[0]['keypoints'].cpu().detach().numpy()
        pts = np.round(pts[0, :, :-1]).astype(int)

        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(np.sqrt(((pts[i] - pts[j]) ** 2).sum()))
        assert all(i > self.min_distance for i in d)

        aimg = align(aimg, pts, self.base_pts, self.dsize)

        return aimg.astype(np.uint8)

    def models_init(self):
        # self.detector = DetectionController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/checkpoints/'
        #             '8/e0659336f7d3449fb7b2eac767df655f/checkpoints/epoch=64-step=16249.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/8/e0659336f7d3449fb7b2eac767df655f/artifacts/mask_rcnn_config.py')
        # ).eval().model_loss
        # self.detector.to(self.device)
        #
        # self.lmd = KeyPointsController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/checkpoints/'
        #             '9/4100f0feaa39434b92b56b5faf000d97/checkpoints/epoch=14-step=62384.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/keypoints_config.py')
        # ).eval().model_loss
        # self.lmd.to(self.device)

        self.detector = DetectionController(get_dict_wrapper(Path('configs/to_reproduce/mask/mask_rcnn_config.py')))
        self.detector.load_state_dict(torch.load(Path('configs/to_reproduce/mask/epoch=64.ckpt')))
        self.detector = self.detector.eval().model_loss
        self.detector.to(self.device)

        self.lmd = KeyPointsController(get_dict_wrapper(Path('configs/to_reproduce/keypoint/keypoints_config.py')))
        self.lmd.load_state_dict(torch.load(Path('configs/to_reproduce/keypoint/epoch=14.ckpt')))
        self.lmd = self.lmd.eval().model_loss
        self.lmd.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector', 'lmd']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()


class IdentityPreproc:
    def __call__(self, img):
        return img


class Preproc13:

    def __init__(self, base_pts, dsize, padding_val, thr=0.9, min_distance=5, device='cpu', old_align=False):
        self.base_pts = base_pts
        self.dsize = dsize
        self.padding_val = padding_val
        self.detector = None
        self.thr = thr
        self.min_distance = min_distance
        self.device = device
        self.old_align = old_align
        self.return_for_metrics = False
        self.models_init()

    @torch.no_grad()
    def __call__(self, img):
        pts = self.detector(torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255)
        score = pts[0]['scores'].cpu().detach().numpy()
        assert len(score) and score[0] > self.thr
        bbox = pts[0]['boxes'].cpu().detach().numpy()[0]
        bbox = np.round(bbox).astype(int)

        if self.return_for_metrics:
            return pts

        aimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return aimg

    def models_init(self):
        # self.detector = KeyPointsController.load_from_checkpoint(
        #     str(
        #         Path(
        #             'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/checkpoints/'
        #             '9/4100f0feaa39434b92b56b5faf000d97/checkpoints/epoch=14-step=62384.ckpt'
        #         )
        #     ),
        #     config=get_dict_wrapper(r'mlruns/9/4100f0feaa39434b92b56b5faf000d97/artifacts/keypoints_config.py')
        # ).eval().model_loss
        self.detector = KeyPointsController(get_dict_wrapper(Path('configs/to_reproduce/keypoint/keypoints_config.py')))
        self.detector.load_state_dict(torch.load(Path('configs/to_reproduce/keypoint/epoch=14.ckpt')))
        self.detector = self.detector.eval().model_loss
        self.detector.to(self.device)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ['detector']}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models_init()
