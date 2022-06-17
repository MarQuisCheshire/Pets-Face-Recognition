import cv2
import numpy as np


def align(img, pts, base_pts, dsize):
    assert len(pts) == len(base_pts)
    if len(pts) == 3:
        pts1 = np.asarray([np.round(np.mean(pts, axis=0)).astype(int)] + pts.tolist())
        pts2 = np.asarray([np.round(np.mean(base_pts, axis=0)).astype(int)] + base_pts.tolist())
    else:
        pts1 = pts
        pts2 = base_pts
    (H, mask) = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    aimg = cv2.warpPerspective(img, H, dsize[:-1][::-1])
    return aimg
