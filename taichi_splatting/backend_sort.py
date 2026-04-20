from functools import cache
from typing import Literal, Tuple

import taichi as ti
import torch

SortBackend = Literal["auto", "cuda", "torch", "taichi_field", "bucket_taichi", "ordered_taichi"]


def _cuda_ops():
  from taichi_splatting import cuda_lib

  return cuda_lib


def _should_use_cuda(backend: SortBackend, tensor: torch.Tensor) -> bool:
  if backend == "cuda":
    return True
  if backend == "torch":
    return False
  return tensor.is_cuda


def full_cumsum(x: torch.Tensor, backend: SortBackend = "auto") -> Tuple[torch.Tensor, int]:
  if backend == "taichi_field":
    return _full_cumsum_taichi(x)
  if _should_use_cuda(backend, x):
    return _cuda_ops().full_cumsum(x)

  if x.shape[0] == 0:
    return x.new_zeros((1, *x.shape[1:])), 0

  try:
    cumsum = torch.cumsum(x, dim=0)
  except RuntimeError:
    cumsum = torch.cumsum(x.cpu(), dim=0).to(device=x.device)

  out = x.new_empty((x.shape[0] + 1, *x.shape[1:]))
  out[0] = 0
  out[1:] = cumsum
  total = int(cumsum[-1].item())
  return out, total


def _sort_key_view(keys: torch.Tensor, start_bit: int = 0, end_bit: int | None = None) -> torch.Tensor:
  sort_keys = keys.to(torch.int64)
  if start_bit == 0 and end_bit is None:
    return sort_keys

  if end_bit is None or end_bit < 0:
    return sort_keys >> start_bit

  width = end_bit - start_bit
  if width <= 0:
    return torch.zeros_like(sort_keys)
  mask = (1 << width) - 1
  return (sort_keys >> start_bit) & mask


def radix_sort_pairs(
  keys: torch.Tensor,
  values: torch.Tensor,
  start_bit: int = 0,
  end_bit: int | None = None,
  backend: SortBackend = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
  if backend == "taichi_field":
    return _radix_sort_pairs_taichi(keys, values, start_bit=start_bit, end_bit=end_bit)
  if _should_use_cuda(backend, keys):
    return _cuda_ops().radix_sort_pairs(keys, values, start_bit=start_bit, end_bit=end_bit)

  sort_keys = _sort_key_view(keys, start_bit=start_bit, end_bit=end_bit)
  try:
    order = torch.argsort(sort_keys, stable=True)
  except RuntimeError:
    order = torch.argsort(sort_keys.cpu(), stable=True).to(device=keys.device)
  return keys[order], values[order]


@cache
def _copy_i32_kernel():
  @ti.kernel
  def kernel(src: ti.types.ndarray(ti.i32, ndim=1), dst: ti.types.ndarray(ti.i32, ndim=1), n: ti.i32):
    for idx in range(n):
      dst[idx] = src[idx]
  return kernel


@cache
def _scan_step_i32_kernel():
  @ti.kernel
  def kernel(
      src: ti.types.ndarray(ti.i32, ndim=1),
      dst: ti.types.ndarray(ti.i32, ndim=1),
      n: ti.i32,
      offset: ti.i32,
  ):
    for idx in range(n):
      value = src[idx]
      if idx >= offset:
        value += src[idx - offset]
      dst[idx] = value
  return kernel


@cache
def _exclusive_from_inclusive_i32_kernel():
  @ti.kernel
  def kernel(
      inclusive: ti.types.ndarray(ti.i32, ndim=1),
      out: ti.types.ndarray(ti.i32, ndim=1),
      n: ti.i32,
  ):
    out[0] = 0
    for idx in range(n):
      out[idx + 1] = inclusive[idx]
  return kernel


@cache
def _sort_stage_u64_i32_kernel():
  @ti.kernel
  def kernel(
      keys: ti.types.ndarray(ti.u64, ndim=1),
      values: ti.types.ndarray(ti.i32, ndim=1),
      n: ti.i32,
      p: ti.i32,
      k: ti.i32,
      invocations: ti.i32,
  ):
    for inv in range(invocations):
      j = k % p + inv * 2 * k
      end = ti.min(k, n - j - k)
      for i in range(end):
        a = i + j
        b = a + k
        if a // (p * 2) == b // (p * 2):
          key_a = keys[a]
          key_b = keys[b]
          if key_a > key_b:
            keys[a] = key_b
            keys[b] = key_a
            temp = values[a]
            values[a] = values[b]
            values[b] = temp
  return kernel


@cache
def _sort_stage_u32_i32_kernel():
  @ti.kernel
  def kernel(
      keys: ti.types.ndarray(ti.u32, ndim=1),
      values: ti.types.ndarray(ti.i32, ndim=1),
      n: ti.i32,
      p: ti.i32,
      k: ti.i32,
      invocations: ti.i32,
  ):
    for inv in range(invocations):
      j = k % p + inv * 2 * k
      end = ti.min(k, n - j - k)
      for i in range(end):
        a = i + j
        b = a + k
        if a // (p * 2) == b // (p * 2):
          key_a = keys[a]
          key_b = keys[b]
          if key_a > key_b:
            keys[a] = key_b
            keys[b] = key_a
            temp = values[a]
            values[a] = values[b]
            values[b] = temp
  return kernel


def _full_cumsum_taichi(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
  if x.ndim != 1 or x.dtype != torch.int32:
    raise NotImplementedError("taichi_field full_cumsum currently supports 1D int32 tensors only.")
  if x.shape[0] == 0:
    return x.new_zeros((1,)), 0

  n = int(x.shape[0])
  work_a = torch.empty_like(x)
  work_b = torch.empty_like(x)
  _copy_i32_kernel()(x.contiguous(), work_a, n)

  src = work_a
  dst = work_b
  offset = 1
  while offset < n:
    _scan_step_i32_kernel()(src, dst, n, offset)
    src, dst = dst, src
    offset *= 2

  out = x.new_empty((n + 1,))
  _exclusive_from_inclusive_i32_kernel()(src, out, n)
  ti.sync()
  return out, int(out[-1].item())


def _radix_sort_pairs_taichi(
    keys: torch.Tensor,
    values: torch.Tensor,
    start_bit: int = 0,
    end_bit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if start_bit != 0:
    raise NotImplementedError("taichi_field radix_sort_pairs currently requires start_bit=0.")
  if keys.ndim != 1 or values.ndim != 1:
    raise NotImplementedError("taichi_field radix_sort_pairs currently supports 1D tensors only.")
  if keys.shape[0] != values.shape[0]:
    raise ValueError(f"keys/values length mismatch: {keys.shape[0]} vs {values.shape[0]}")
  if values.dtype != torch.int32:
    raise NotImplementedError("taichi_field radix_sort_pairs currently requires int32 values.")
  if keys.dtype not in {torch.uint32, torch.uint64}:
    raise NotImplementedError("taichi_field radix_sort_pairs currently requires uint32 or uint64 keys.")

  n = int(keys.shape[0])
  sorted_keys = keys.contiguous().clone()
  sorted_values = values.contiguous().clone()
  if n <= 1:
    return sorted_keys, sorted_values

  sort_stage = _sort_stage_u64_i32_kernel() if keys.dtype == torch.uint64 else _sort_stage_u32_i32_kernel()
  p = 1
  while p < n:
    k = p
    while k >= 1:
      invocations = int((n - k - k % p) / (2 * k)) + 1
      sort_stage(sorted_keys, sorted_values, n, p, k, invocations)
      ti.sync()
      k //= 2
    p *= 2
  return sorted_keys, sorted_values
