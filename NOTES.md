# Taichi Mac Splatting Notes

This file is for future agents and maintainers trying to improve the Mac /
Apple Silicon path. It records what we tried, what worked, what failed, and what
should be tested next.

## Current Best Mac Path

Use:

```text
kernel_variant="metal_reference"
sort_backend="auto"
backward_variant="pixel_reference"
metal_compatible=True
```

Why:

- `metal_reference` avoids CUDA-only Taichi SIMT block/warp intrinsics and
  compiles on `ti.metal`.
- `sort_backend="auto"` uses the fastest available non-CUDA fallback on Mac,
  currently PyTorch/MPS ordering.
- `pixel_reference` backward matched a direct Torch packed-2D baseline in small
  forward/backward checks.

This is a correctness and baseline-throughput path, not a final high-performance
renderer.

## What We Tried

### CUDA Path On Mac

Status:
    Not viable.

Reason:
    Upstream Taichi Splatting assumes CUDA helpers for sort/cumsum and uses
    CUDA-oriented Taichi SIMT/block/warp intrinsics in the fast raster path.
    Generic Taichi has a Metal backend, but this package was not just missing a
    small backend flag.

Decision:
    Keep CUDA behavior intact where possible, but add a separate Metal-compatible
    reference path.

### Torch/MPS Sort Fallback

Status:
    Fastest working Mac path in this fork.

What it does:
    Uses PyTorch/MPS operations for tile/depth ordering fallback, then uses
    Taichi/Metal for raster forward/backward.

Why it won:
    It avoids slow custom Taichi global sort kernels on Metal and lets PyTorch
    handle the ordering stage.

Limit:
    This still uses separate ordering and raster stages. It is not a fused
    tile-local sort/raster pipeline.

### `taichi_field` Global Sort

Status:
    Correct, too slow.

What it tried:
    A Taichi-native global ndarray scan/sort reference path to avoid CUDA.

Observed behavior:
    Correct on small checks, but far slower than the PyTorch/MPS sort fallback.

Lesson:
    "Move sorting into Taichi" is not sufficient. The algorithm and memory
    traffic dominate.

### `bucket_taichi` Compact Per-Tile Bucket Sort

Status:
    Correct, still slower than the reference path.

What it tried:
    Count overlaps per tile, prefix-sum tile counts, fill compact tile buckets
    with atomics, then sort within each tile range.

Observed behavior:
    It matched small accuracy checks and was faster than the global Taichi sort,
    but still lost to the PyTorch/MPS sort reference.

Why it probably lost:
    The implementation stages binning, sorting, and rasterization through
    separate kernels and global memory. The intended high-performance design is
    to load one tile's bucket into threadgroup memory, sort locally, and
    immediately rasterize without writing a sorted list back to global memory.

### `ordered_taichi`

Status:
    Only correct for pre-sorted inputs, not a general renderer.

What it tried:
    Preserve caller-provided input order within every tile.

When it is correct:
    Only when the caller already supplies splats in front-to-back depth order.

Why it is not enough:
    The naive tile-major scan is expensive, and the correctness contract is too
    narrow for general use.

### Depth16 / Low Precision

Status:
    Not a real speed path yet.

Reason:
    The current tile mapper imports f32 `Gaussian2D` and expects f32 depths.
    Low precision needs separate mapper/raster kernels and careful key packing,
    not just casting tensors to float16.

### Per-Splat Backward

Status:
    Not implemented in this fork.

Expected value:
    Could reduce saved memory if paired with recompute-style backward.

Caveat:
    It should probably be designed together with the fused tile-local renderer,
    because the current separated reference path is not the target architecture.

## Accuracy Status

Small packed-2D checks against a direct CPU-float64 Torch reference passed for:

- forward image
- packed Gaussian gradients
- feature gradients

Tested cases included 16/32px images and 4/8/16 splats, with absolute error
around `1e-8` in the observed run.

What this proves:
    The simple Metal-compatible raster/backward math is credible.

What it does not prove:
    Full 3D projection correctness, real-scene quality, or high performance.

## Performance Snapshot

Dynaworld synthetic random splat probes observed:

```text
128x128,  G=65536, metal_reference auto sort: about 25.7 ms forward
1024x1024, G=65536, metal_reference auto sort: about 36.4 ms forward
```

Older broader sweeps showed 131k and 262k splats were below 60 FPS.

These timings should be treated as local Apple Silicon synthetic probes. They
are useful for comparing code paths, not as final product benchmarks.

## Sorting / Key-Pressure Findings

We added overlap-key pressure diagnostics in Dynaworld to measure:

- `K`: total Gaussian-tile overlap keys
- `K/G`: duplicate tile keys per splat
- p95/max tiles per splat
- p95/max splats per tile

Random-splat findings:

```text
64x64,    G=1024:  taichi_obb / exact_conic = 1.090x keys
128x128,  G=65536: taichi_obb / exact_conic = 1.093x keys
1024x1024,G=65536: taichi_obb / exact_conic = 1.099x keys
```

Interpretation:
    For random splats, Taichi's current OBB tile query is already close to exact
    conic/tile intersection. AccuTile-style exactness is probably not the main
    speed unlock for this fork on synthetic data.

Open caveat:
    Real trained outputs may produce large transparent floaters. Those could
    make `K/G` much worse than random splats. Always measure trained-scene `K/G`
    before deciding that culling is unimportant.

## What Future Agents Should Do Next

### 1. Measure Real Training Outputs

Before changing the renderer, run a short real training job with overlap
diagnostics enabled in Dynaworld:

```jsonc
"logging": {
  "with_metrics": {
    "renderer": true,
    "optimizer": false,
    "every": 25,
    "print_summary": true,
    "wandb": true,
    "fail_fast": true
  }
}
```

Watch:

```text
TileDiag/ExactConic_duplication_factor_mean
TileDiag/ExactConic_max_tiles_per_splat_max
TileDiag/ExactConic_max_splats_per_tile_max
TileDiag/CustomRectToExactKeyRatio
```

If trained `K/G` is low:
    Stop chasing culling and build a fused Metal renderer.

If trained `K/G` is high:
    Work on floaters, pruning, projected-radius limits, or exact tile
    intersection before more sort architecture work.

### 2. Build The Actual Fast Renderer Shape

The speed target is:

```text
count/bin -> prefix/cursors -> fused tile-local sort + raster -> recompute backward
```

Key properties:

- sort per tile in threadgroup memory
- fuse local sort and raster
- avoid writing sorted per-tile arrays back to global memory
- recompute alpha/transmittance in backward
- save compact tile bins, not dense pixel activations

This may be hard to express cleanly in Taichi. A raw Metal / MLX / Torch native
extension may be the right final backend, with this fork kept as a correctness
and integration baseline.

### 3. Only Port AccuTile If The Metrics Justify It

Do not port AccuTile just because sorting is slow.

Port exact AccuTile-style row/column tile intersection if same-tile exact
intersection saves a large fraction of keys on target trained outputs, e.g.
more than about 25-30%.

If exact saves only about 10%, it is likely a cleanup, not the main speed
project.

## Known Limitations

- No final high-performance fused Metal renderer.
- No optimized low-precision path.
- No per-splat backward memory optimization.
- No overflow handling for huge per-tile splat lists.
- No differentiation through sort order, tile support, or visibility threshold.
- Benchmarks here are synthetic unless otherwise stated.

## Useful Search Terms

- Taichi Mac Splatting
- Taichi gsplat Mac
- Taichi Gaussian Splatting Mac
- Taichi Gaussian Splat renderer Mac
- differentiable Gaussian splatting renderer Mac
- fast Gaussian splatting differentiable renderer Mac
- Apple Silicon Gaussian Splatting PyTorch MPS Metal
