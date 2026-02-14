from pathlib import Path
from typing import Callable

import cupy as cp
import cv2 as cv
import numpy as np

from ..lib.convolve import scharr_x, scharr_y
from ..lib.pyramid import build_pyramid
from ..lib.lucas_kanade import lucas_kanade

UpdateFrame = Callable[[np.ndarray], np.ndarray]

def show_image(image: np.ndarray):
  WINDOW_TITLE = 'Video'
  WINDOW_WIDTH = 1600
  WINDOW_HEIGHT = 1200
  cv.namedWindow(WINDOW_TITLE, cv.WINDOW_NORMAL)
  cv.resizeWindow(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
  cv.imshow(WINDOW_TITLE, image)
  cv.waitKey(0)
  cv.destroyAllWindows()


def show_gradients(file: Path):
  im = cv.imread(str(file))
  gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
  gray_gpu = cp.asarray(gray, dtype=cp.float32)
  dx = scharr_x(gray_gpu)
  dy = scharr_y(gray_gpu)
  result = cp.hstack((dx, dy)).astype(cp.uint8).get()
  show_image(result)


def show_pyramid(file: Path):
  im = cv.imread(str(file))
  gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
  gray_gpu = cp.asarray(gray, dtype=cp.float32)
  height, width = gray_gpu.shape
  pyramid = build_pyramid(gray_gpu, 2)
  pads = [((0, height - level.shape[0]), (0, width - level.shape[1])) for level in pyramid[1:]]
  pyramid = [pyramid[0]] + [
    cp.pad(level, pad, mode='constant', constant_values=255.0) for level, pad in zip(pyramid[1:], pads, strict=True)
  ]
  result = cp.hstack(pyramid).astype(cp.uint8).get()
  show_image(result)


def show_lucas_kanade(
  previous: Path,
  next: Path,
  num_points: int,
  window_size: tuple[int, int],
  max_levels: int,
  max_iterations: int,
  threshold: float,
):
  previous_image = cv.imread(str(previous))
  next_image = cv.imread(str(next))
  previous_gray = cv.cvtColor(previous_image, cv.COLOR_BGR2GRAY)
  next_gray = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
  previous_gpu = cp.asarray(previous_gray, dtype=cp.float32)
  next_gpu = cp.asarray(next_gray, dtype=cp.float32)

  points = cv.goodFeaturesToTrack(previous_gray, maxCorners=num_points, qualityLevel=0.01, minDistance=10).reshape(-1, 2)
  previous_points = cp.asarray(points, dtype=cp.int32)
  new_points, status = lucas_kanade(previous_gpu, next_gpu, previous_points, window_size, max_levels, max_iterations, threshold)

  colors = np.random.randint(0, 255, (num_points, 3))
  new_points = new_points.get().astype(np.int32)
  status = status.get()

  good_previous = points[status == 1]
  good_new = new_points[status == 1]
  colors = colors[status == 1]

  old_mask = np.zeros_like(previous_image)
  new_mask = np.zeros_like(next_image)
  
  for prev, new, color in zip(good_previous, good_new, colors, strict=True):
    old_mask = cv.circle(old_mask, tuple(prev.astype(int)), 10, tuple(color.tolist()), -1)
    new_mask = cv.circle(new_mask, tuple(new.astype(int)), 10, tuple(color.tolist()), -1)

  old_result = cv.add(previous_image, old_mask)
  new_result = cv.add(next_image, new_mask)
  result = np.hstack([old_result, new_result])
  show_image(result)
