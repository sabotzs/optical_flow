import cupy as cp
from cupyx.scipy.ndimage import convolve


def scharr_x(image: cp.ndarray) -> cp.ndarray:
  SCHARR_KERNEL_X = cp.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=cp.float32)
  return convolve(image, SCHARR_KERNEL_X, mode='reflect')


def scharr_y(image: cp.ndarray) -> cp.ndarray:
  SCHARR_KERNEL_Y = cp.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=cp.float32)
  return convolve(image, SCHARR_KERNEL_Y, mode='reflect')


def pyramid_convolution(image: cp.ndarray) -> cp.ndarray:
  INV_256 = 0.00390625  # 1 / 256
  # fmt: off
  PYRAMID_KERNEL = INV_256 * cp.array(
    [
      [1, 4, 6, 4, 1],
      [4, 16, 24, 16, 4],
      [6, 24, 36, 24, 6],
      [4, 16, 24, 16, 4],
      [1, 4, 6, 4, 1]
    ],
    dtype=cp.float32
  )
  # fmt: on
  return convolve(image, PYRAMID_KERNEL, mode='reflect')[::2, ::2]
