import cupy as cp

from .convolve import pyramid_convolution


def build_pyramid(image: cp.ndarray, levels: int) -> list[cp.ndarray]:
  pyramid = [image]
  for _ in range(1, levels + 1):
    pyramid.append(pyramid_convolution(pyramid[-1]))
  return pyramid
