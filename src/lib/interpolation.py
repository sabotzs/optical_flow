import cupy as cp


def bilinear_interpolation(image: cp.ndarray, coords: cp.ndarray) -> cp.ndarray:
  x = coords[:, 0]
  y = coords[:, 1]

  x0 = cp.clip(cp.floor(x).astype(cp.int32), 0, image.shape[1] - 1)
  y0 = cp.clip(cp.floor(y).astype(cp.int32), 0, image.shape[0] - 1)
  x1 = cp.clip(x0 + 1, 0, image.shape[1] - 1)
  y1 = cp.clip(y0 + 1, 0, image.shape[0] - 1)

  Ia = image[y0, x0]
  Ib = image[y0, x1]
  Ic = image[y1, x0]
  Id = image[y1, x1]

  dx = x - x0
  dy = y - y0
  r1 = Ia + dy * (Ic - Ia)
  r2 = Ib + dy * (Id - Ib)
  return r1 + dx * (r2 - r1)
