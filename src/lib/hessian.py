import cupy as cp


def calculate_inv_hessians(jacobian: cp.ndarray) -> cp.ndarray:
  points_count = jacobian.shape[0]
  nabla = jacobian.reshape(points_count, -1, 6)
  hessians = cp.einsum('npi,npj->nij', nabla, nabla)
  return cp.linalg.inv(hessians)
