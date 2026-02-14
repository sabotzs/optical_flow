import cupy as cp


def calculate_jacobians(
  I_x: cp.ndarray,
  I_y: cp.ndarray,
  x_grid: cp.ndarray,
  y_grid: cp.ndarray,
) -> cp.ndarray:
  return cp.stack(
    [
      I_x,
      I_y,
      I_x * x_grid,
      I_y * x_grid,
      I_x * y_grid,
      I_y * y_grid,
    ],
    axis=-1,
  )
