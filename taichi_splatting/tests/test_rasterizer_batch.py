import taichi as ti
import torch

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import rasterize, rasterize_batch
from taichi_splatting.taichi_queue import taichi_queue


def _make_inputs(device=torch.device("cpu")):
  torch.manual_seed(0)
  batch_size = 2
  point_count = 5
  feature_size = 3
  image_size = (16, 16)

  means = torch.rand(batch_size, point_count, 2, device=device) * torch.tensor(image_size, device=device)
  axis = torch.randn(batch_size, point_count, 2, device=device)
  axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  sigma = torch.rand(batch_size, point_count, 2, device=device) * 2.0 + 1.0
  alpha = torch.rand(batch_size, point_count, 1, device=device) * 0.5 + 0.1
  packed = torch.cat([means, axis, sigma, alpha], dim=-1).float()

  depth = torch.linspace(0.0, 1.0, point_count, device=device).view(1, point_count, 1)
  depth = depth.expand(batch_size, -1, -1).contiguous()
  features = torch.rand(batch_size, point_count, feature_size, device=device).float()
  return packed, depth, features, image_size


def test_rasterize_batch_matches_loop_forward_backward_cpu():
  config = RasterConfig(
    tile_size=8,
    metal_compatible=True,
    kernel_variant="metal_reference",
    sort_backend="torch",
    backward_variant="pixel_reference",
  )

  with taichi_queue(arch=ti.cpu, log_level=ti.ERROR):
    packed, depth, features, image_size = _make_inputs()

    packed_batch = packed.clone().requires_grad_(True)
    features_batch = features.clone().requires_grad_(True)
    batch = rasterize_batch(packed_batch, depth, features_batch, image_size, config)
    batch_loss = batch.image.square().mean() + batch.image_weight.square().mean()
    batch_loss.backward()

    packed_loop = packed.clone().requires_grad_(True)
    features_loop = features.clone().requires_grad_(True)
    loop_images = []
    loop_weights = []
    for batch_idx in range(packed.shape[0]):
      out = rasterize(packed_loop[batch_idx], depth[batch_idx], features_loop[batch_idx], image_size, config)
      loop_images.append(out.image)
      loop_weights.append(out.image_weight)

    loop_image = torch.stack(loop_images, dim=0)
    loop_weight = torch.stack(loop_weights, dim=0)
    loop_loss = loop_image.square().mean() + loop_weight.square().mean()
    loop_loss.backward()

  assert torch.allclose(batch.image, loop_image, atol=0.0, rtol=0.0)
  assert torch.allclose(batch.image_weight, loop_weight, atol=0.0, rtol=0.0)
  assert torch.allclose(packed_batch.grad, packed_loop.grad, atol=1e-6, rtol=1e-5)
  assert torch.allclose(features_batch.grad, features_loop.grad, atol=1e-6, rtol=1e-5)
