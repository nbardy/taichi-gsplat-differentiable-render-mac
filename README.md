# Taichi Mac Splatting

Fast splatting on the Mac integrated with Taichi.

Taichi Gaussian Splat Renderer for Mac, Apple Silicon, MPS, and Metal.

## Which Mac Renderer Should You Use?

Use this repository when you need Taichi compatibility or want a Taichi-native
renderer path that works with PyTorch tensors on Apple Silicon.

If you only want the fastest Mac raster hot path and do not need Taichi, use
[`fast-mac-gsplat`](https://github.com/nbardy/fast-mac-gsplat). It is the
recommended high-performance path: a blazing fast pure Metal/Torch extension
with local tile sorting, recompute backward, and much higher throughput than
this Taichi compatibility fork.

## Import And Run The Fastest Mac Path

From this repository checkout:

```bash
python -m pip install -e .
```

The fastest working Mac path in this fork is still the reference hybrid path:

```text
kernel_variant="metal_reference"
sort_backend="auto"
backward_variant="pixel_reference"
metal_compatible=True
```

On Apple Silicon this means:

* PyTorch/MPS handles the tile/depth ordering fallback.
* Taichi/Metal handles raster forward and backward.
* This is the path to use first for correctness and baseline Mac throughput.

Minimal projected-2D raster usage:

```python
import taichi as ti
import torch

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import rasterize, rasterize_batch
from taichi_splatting.taichi_queue import TaichiQueue

device = torch.device("mps")
TaichiQueue.init(arch=ti.metal, log_level=ti.ERROR)

config = RasterConfig(
    tile_size=16,
    alpha_threshold=1.0 / 255.0,
    clamp_max_alpha=0.99,
    saturate_threshold=0.9999,
    metal_compatible=True,
    kernel_variant="metal_reference",
    sort_backend="auto",
    backward_variant="pixel_reference",
)

G = 1024
height, width = 128, 128

# Packed 2D Gaussian rows:
# [mean_x, mean_y, axis_x, axis_y, sigma_x, sigma_y, opacity]
means = torch.rand((G, 2), device=device, dtype=torch.float32) * torch.tensor([width, height], device=device)
axis = torch.nn.functional.normalize(torch.randn((G, 2), device=device, dtype=torch.float32), dim=-1)
sigma = torch.rand((G, 2), device=device, dtype=torch.float32) * 4.0 + 1.0
opacity = torch.rand((G, 1), device=device, dtype=torch.float32) * 0.5 + 0.1
gaussians2d = torch.cat([means, axis, sigma, opacity], dim=-1).contiguous()
depth = torch.linspace(0.0, 1.0, G, device=device).view(-1, 1)
features = torch.rand((G, 3), device=device, dtype=torch.float32)

raster = rasterize(
    gaussians2d,
    depth,
    features,
    image_size=(width, height),
    config=config,
)

background = torch.ones(3, device=device)
image = raster.image + (1.0 - raster.image_weight).clamp(0.0, 1.0).unsqueeze(-1) * background
loss = image.square().mean()
loss.backward()
```

For native batched rendering, use `rasterize_batch` with batched tensors:

```python
# gaussians2d: [B, G, 7], depth: [B, G, 1], features: [B, G, C]
raster = rasterize_batch(
    gaussians2d,
    depth,
    features,
    image_size=(width, height),
    config=config,
)

# raster.image: [B, H, W, C], raster.image_weight: [B, H, W]
```

The batch path uses real batch/tile keys and batched raster kernels. It does not
pack frames into an atlas.

For full 3D projection and benchmark examples, see the Dynaworld benchmark
harness that vendors this fork:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_taichi_metal_scale.jsonc
```

## Fastest Final Versions We Measured

These are local Apple Silicon benchmark results from Dynaworld synthetic random
splat probes. Treat them as directionally useful, not as final package claims.

| Path | Status | Notes |
| --- | --- | --- |
| `metal_reference` + `sort_backend="auto"` | fastest Taichi/Mac path in this fork | Uses PyTorch/MPS ordering plus Taichi/Metal raster/backward. |
| `bucket_taichi` | correct but slower | Per-tile bucket sort is the right high-level idea, but this implementation uses separate global-memory stages. |
| `taichi_field` | correct but too slow | Global Taichi ndarray sort is a reference path, not a speed path. |
| `ordered_taichi` | narrow correctness only | Preserves input order; only correct when caller pre-sorts splats front-to-back. |

Representative forward-only synthetic probes:

| Resolution | Splats | Fastest Taichi path |
| --- | ---: | ---: |
| 128x128 | 65,536 | about 25.7 ms |
| 1024x1024 | 65,536 | about 36.4 ms |

Small packed-2D accuracy checks matched a direct CPU-float64 Torch reference for
forward and backward gradients at about `1e-8` absolute error in the tested
16/32px cases with 4/8/16 splats.

Native batch also matches a direct packed-2D Torch reference on MPS in small
batched checks. A local Apple Silicon probe compared `rasterize_batch` against a
direct Torch implementation of the same packed 2D Gaussian math:

| Case | Taichi native batch | Direct Torch | Speedup |
| --- | ---: | ---: | ---: |
| `B=4`, 64x64, 128 splats, forward | `22.268 ms` | `169.574 ms` | `7.6x` |
| `B=4`, 64x64, 128 splats, forward+backward | `46.894 ms` | `791.411 ms` | `16.9x` |
| `B=4`, 128x128, 128 splats, forward | `21.665 ms` | `667.194 ms` | `30.8x` |
| `B=4`, 128x128, 128 splats, forward+backward | `37.359 ms` | `1350.545 ms` | `36.2x` |

Forward agreement for those checks was about `1e-7` max absolute error. The
direct Torch reference gets impractically slow at larger image/splat counts.

For comparison, the pure Metal
[`fast-mac-gsplat`](https://github.com/nbardy/fast-mac-gsplat) repo reports
`128x128 / 512` direct-Torch speedups around `22-27x` forward and `73-93x`
forward+backward, plus 4K / 65,536-splat hot-path timings around `12-14 ms`
forward for its v3 path. Use that project when maximum speed matters more than
Taichi integration.

The 4K / 65,536-splat pure-Metal numbers are recorded explicitly in that repo:

| Renderer | 4096x4096 / 65,536 splats | Forward | Forward+backward |
| --- | --- | ---: | ---: |
| `fast-mac-gsplat` v3 | sigma 1-5 px | `12.410 ms` | `47.872 ms` |
| `fast-mac-gsplat` v3 | sigma 3-8 px | `13.702 ms` | `60.738 ms` |

There is no direct Torch number for 4K / 65,536 splats because it is not a
useful baseline. A dense vectorized Torch renderer would need to materialize
roughly `4096 * 4096 * 65536 = 1.1e12` pixel-splat evaluations, which is
terabytes of activation traffic before backward. A looped Torch reference avoids
that one giant tensor but is still far too slow for an interactive benchmark.

For detailed experiment notes and future directions, read [NOTES.md](NOTES.md).

## Search / Fork Positioning

Experimental Apple Silicon fork of Taichi Splatting for differentiable Gaussian
splat rasterization from PyTorch on MPS/Metal. This repository is intended to be
findable for searches like:

* Taichi gsplat Mac
* Taichi Gaussian Splatting Mac
* Taichi Gaussian Splat renderer Mac
* differentiable Gaussian splatting renderer Mac
* fast Gaussian splatting differentiable renderer Mac
* Apple Silicon Gaussian Splatting PyTorch MPS Metal

## Upstream Taichi Splatting

Rasterizer for Gaussian Splatting using Taichi and PyTorch - embedded in a Python library. Currently very usable but in active development, so likely will break with new versions!

Trainer: [here](https://github.com/uc-vision/splat-trainer)
Viewer: [here](https://github.com/uc-vision/splat-trainer)

This work is originally derived off [Taichi 3D Gaussian Splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting), with significant re-organisation and changes.

Key differences are the rendering algorithm is decomposed into separate operations (projection, shading functions, tile mapping and rasterization) which can be combined in different ways in order to facilitate a more flexible use, and gradients can be enabled on "all the things" as required for the application (and not when disabled, to save performance).

Using the Taichi autodiff for a simpler implementation where possible (e.g. for projection, but not for the rasterization).

## Mac / Metal fork status

This fork carries an experimental Apple Silicon path for differentiable 2D
Gaussian splat rasterization. The goal is to make the rasterizer usable from
PyTorch on Macs while preserving the CUDA path for machines that have it.

Current fork changes:

* Adds non-CUDA sort/cumsum fallbacks behind `sort_backend` so the tile mapper
  can run on MPS/Metal without `taichi_splatting.cuda_lib`.
* Adds a `metal_reference` rasterizer variant that avoids CUDA-only Taichi SIMT
  block/warp intrinsics and compiles on `ti.metal`.
* Adds a `pixel_reference` backward path for Metal that matches a direct Torch
  packed-2D Gaussian baseline in small forward/backward gradient checks.
* Adds experimental Taichi-side sort modes:
  * `taichi_field`: global Taichi ndarray scan/sort reference. Correct, but
    slow.
  * `bucket_taichi`: compact per-tile bucket sort. Correct in small gradient
    checks and faster than the global Taichi sort, but still slower than the
    current reference path.
  * `ordered_taichi`: preserves input order inside each tile. Only correct when
    callers already provide front-to-back sorted splats.

Known limitations:

* The fastest Mac path is still the reference path that uses PyTorch/MPS for
  tile/depth ordering and Taichi/Metal for raster/backward.
* The Taichi-native sort experiments are not performance wins yet. The next real
  speed target is a fused tile-local sort and render kernel using Metal
  threadgroup memory.
* Low precision and per-splat backward are not implemented in the Metal path.

Examples:
  * Projecting features for lifting 2D to 3D
  * Colours via. spherical harmonics
  * Depth covariance without needing to build it into the renderer and remaining differentiable.
  * Fully differentiable camera parameters (and ability to swap in new camera models)

## Performance

A document describing some performance benchmarks of taichi-splatting [here](BENCHMARK.md). Through various optimizations, in particular optimizing the summation of gradients in the backward gradient kernel. Taichi-splatting achieves a very large speedup (often an order of magnitude) over the original taichi_3d_gaussian_splatting, and is faster than the reference diff_guassian_rasterization for a complete optimization pass (forward+backward), in particular much faster at higher resolutions.


## Installing

### External dependencies
Create an environment (for example conda with mambaforge) with the following dependencies:

* python >= 3.10
* pytorch - from either conda  Follow instructions [https://pytorch.org/](here).
* taichi-nightly `pip install --upgrade -i https://pypi.taichi.graphics/simple/ taichi-nightly`

### Install

One of:
* `pip install taichi-splatting`
* Clone down with `git clone` and install with `pip install ./taichi-splatting`

## Executables

### fit_image_gaussians

There exists a toy optimizer for fitting a set of randomly initialized gaussians to some 2D images `fit_image_gaussians` - useful for testing rasterization without the rest of the dependencies.

Fitting an image (fixed points): \
`fit_image_gaussians <image file> --show  --n 20000` 

Fitting an image (split and prune to target): \
`fit_image_gaussians <image file> --show --n 1000 --target 20000` 

See `--help` for other options.

### benchmarks

There exist benchmarks to evaluate performance on individual components in isolation under `taichi_splatting/benchmarks/`

### tests 

Tests (gradient tests and tests comparing to torch-based reference implementations) can be run with pytest, or individually under 
`taichi_splatting/tests/`

### splat-viewer

A viewer for reconstructions created with the original gaussian-splatting repository can be found [here](https://github.com/uc-vision/splat-viewer) or installed with pip. Has dependencies on open3d and Qt. 

### splat-benchmark

A benchmark for a full rendererer (in the same repository as above) with real reconstructions (rendering the original camera viewpoints).  Options exist for tweaking all the renderer parameters, benchmarking backward pass etc.


## Progress

### Done
* Benchmarks with original + taichi_3dgs rasterizer

* Simple view culling 
* Projection with autograd
* Tile mapping (optimized and improved culling) 
* Rasterizer forward pass and optimized backward pass

* Spherical harmonics with autograd
* Gradient tests for most parts (float64) - including rasterizer!
* Fit to image training example/test
* Depth and depth-covariance rendering

* Compute point visibility in backward pass (useful for model pruning)
* Example training on images with split/prune operations
* Novel heuristics for split and prune operations computed optionally in backward pass



### Todo

* Backward projection autograd takes a while to compile and is not cached properly
* 16 bit representations of parameters
* Depth rendering/regularization method (e.g. 2DGS or related method)
* Some ideas for optimized tilemapper with flat representations (no inner loop)


### Improvements

* Exposed all internal constants as parameters
* Switched to matrices as inputs instead of quaternions
* Tile mapping tighter culling for tile overlaps (~30% less rendered splats!)
* All configuration parameters exposed (e.g. tile_size, saturation threshold etc.)
* Warp reduction based backward pass for rasterizer, a decent boost in performance


## Conventions

### Transformation matrices

Transformations are notated `T_x_y`, for example `T_camera_world` can be used to transform points in the world to points in the local camera by `points_camera = T_camera_world @ points_world`

