import cupy as cp

from .convolve import scharr_x, scharr_y
from .hessian import calculate_inv_hessians
from .interpolation import bilinear_interpolation
from .jacobian import calculate_jacobians
from .pyramid import build_pyramid


def lucas_kanade(
  previous: cp.ndarray,
  next: cp.ndarray,
  points: cp.ndarray,
  window_size: tuple[int, int],
  max_levels: int,
  max_iterations: int,
  threshold: float,
) -> tuple[cp.ndarray, cp.ndarray]:
  height, width = previous.shape
  n_points = len(points)
  window_height, window_width = window_size
  w_x = (window_width - 1) >> 1
  w_y = (window_height - 1) >> 1
  pad = ((w_y, w_y), (w_x, w_x))

  I_pyramid = [cp.pad(level, pad, mode='reflect') for level in build_pyramid(previous, max_levels)]
  I_x_pyramid = [scharr_x(level) for level in I_pyramid]
  I_y_pyramid = [scharr_y(level) for level in I_pyramid]
  J_pyramid = build_pyramid(next, max_levels)

  xs = cp.arange(-w_x, w_x + 1, dtype=cp.float32)
  ys = cp.arange(-w_y, w_y + 1, dtype=cp.float32)
  x_grid, y_grid = cp.meshgrid(xs, ys)
  coords = cp.stack([x_grid.ravel(), y_grid.ravel()], axis=0)

  As = cp.tile(cp.eye(2, dtype=cp.float32), (n_points, 1, 1))
  vs = (points >> (max_levels + 1)).astype(cp.float32)
  converged = cp.zeros(n_points, dtype=cp.uint8)
  threshold_sq = threshold * threshold

  for level in reversed(range(max_levels + 1)):
    us = (points >> level).astype(cp.int32)
    pus = us + cp.array([[w_x, w_y]], dtype=cp.int32)
    py_indices = pus[:, 1, None] + ys.astype(cp.int32)
    px_indices = pus[:, 0, None] + xs.astype(cp.int32)

    I_comp = I_pyramid[level][py_indices[:, :, None], px_indices[:, None, :]]
    I_x = I_x_pyramid[level][py_indices[:, :, None], px_indices[:, None, :]]
    I_y = I_y_pyramid[level][py_indices[:, :, None], px_indices[:, None, :]]
    J = J_pyramid[level]

    jacobians = calculate_jacobians(I_x, I_y, x_grid, y_grid)
    inv_hessians = calculate_inv_hessians(jacobians)
    vs = 2 * vs

    for _ in range(max_iterations):
      warped_coords = cp.einsum('nij,jk->nik', As, coords) + vs[:, :, None]
      J_comp = bilinear_interpolation(J, warped_coords).reshape(n_points, window_height, window_width)

      I_t = I_comp - J_comp
      b = cp.einsum('nhwj,nhw->nj', jacobians, I_t)
      ni = cp.einsum('nij,nj->ni', inv_hessians, b)
      vs = cp.einsum('nij,nj->ni', As, ni[:, :2]) + vs
      delta_A = cp.eye(2, dtype=cp.float32) + ni[:, 2:].reshape(n_points, 2, 2).transpose(0, 2, 1)
      As = cp.einsum('nij,njk->nik', As, delta_A)

      ni_norm_sq = (ni * ni).sum(axis=1)
      newly_converged = ni_norm_sq < threshold_sq
      converged = converged | newly_converged
      if cp.all(converged):
        break

  valid_bounds = (vs[:, 0] >= 0) & (vs[:, 0] < width) & (vs[:, 1] >= 0) & (vs[:, 1] < height)
  status = (converged & valid_bounds).astype(cp.int8)

  return vs, status
