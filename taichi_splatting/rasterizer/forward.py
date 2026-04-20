from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.concurrent import warp_add_vector_32, warp_add_vector_64


@cache
def forward_kernel(config: RasterConfig, feature_size: int, dtype=ti.f32):
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D
  vec1 = lib.vec1

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size
  kernel_variant = config.kernel_variant
  if kernel_variant == "auto":
    kernel_variant = "metal_reference" if config.metal_compatible else "cuda_simt"

  pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64

  if kernel_variant == "metal_reference":
    @ti.kernel
    def _forward_kernel_simple(
        # Input tensors
        points: ti.types.ndarray(Gaussian2D.vec, ndim=1),              # [N, 7] 2D gaussian parameters
        point_features: ti.types.ndarray(feature_vec, ndim=1),         # [N, F] gaussian features

        # Tile data structures
        tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [T] start/end range of overlapping points
        overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to point index

        # Output image buffers
        image_feature: ti.types.ndarray(feature_vec, ndim=2),          # [H, W, F] output features
        image_alpha: ti.types.ndarray(dtype, ndim=2),                  # [H, W] output alpha

        # Output visibility buffer
        visibility: ti.types.ndarray(dtype, ndim=1)                    # [N] visibility per point
    ):
      camera_height, camera_width = image_feature.shape
      tiles_wide = (camera_width + tile_size - 1) // tile_size

      if ti.static(config.metal_block_dim > 0):
        ti.loop_config(block_dim=config.metal_block_dim)
      for y, x in image_feature:
        tile_x = x // tile_size
        tile_y = y // tile_size
        tile_id = tile_x + tile_y * tiles_wide
        start_offset, end_offset = tile_overlap_ranges[tile_id]

        pixelf = ti.Vector([ti.cast(x, dtype) + 0.5, ti.cast(y, dtype) + 0.5])
        accum_features = feature_vec(0.0)
        total_weight = dtype(0.0)

        for overlap_idx in range(start_offset, end_offset):
          point_idx = overlap_to_point[overlap_idx]
          mean, axis, sigma, point_alpha = Gaussian2D.unpack(points[point_idx])

          gaussian_alpha = pdf(pixelf, mean, axis, sigma)
          alpha = point_alpha * gaussian_alpha
          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

          if alpha > ti.static(config.alpha_threshold) and total_weight < ti.static(config.saturate_threshold):
            weight = alpha * (1.0 - total_weight)
            total_weight += weight

            if ti.static(config.use_alpha_blending):
              accum_features += point_features[point_idx] * weight
            else:
              if total_weight >= ti.static(config.median_threshold):
                accum_features = point_features[point_idx]

            if ti.static(config.compute_visibility):
              ti.atomic_add(visibility[point_idx], weight)

        image_feature[y, x] = accum_features
        image_alpha[y, x] = total_weight

    return _forward_kernel_simple

  if kernel_variant != "cuda_simt":
    raise ValueError(f"Unsupported rasterizer forward kernel_variant={kernel_variant!r}")

  @ti.kernel
  def _forward_kernel(
      # Input tensors
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),              # [N, 7] 2D gaussian parameters
      point_features: ti.types.ndarray(feature_vec, ndim=1),         # [N, F] gaussian features

      # Tile data structures
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [T] start/end range of overlapping points
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to point index

      # Output image buffers
      image_feature: ti.types.ndarray(feature_vec, ndim=2),          # [H, W, F] output features
      image_alpha: ti.types.ndarray(dtype, ndim=2),                  # [H, W] output alpha

      # Output visibility buffer
      visibility: ti.types.ndarray(dtype, ndim=1)                    # [N] visibility per point (if compute_visibility is True) 
  ):
    camera_height, camera_width = image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size
    tiles_high = (camera_height + tile_size - 1) // tile_size

    ti.loop_config(block_dim=(tile_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):
      pixel = tiling.tile_transform(tile_id, tile_idx, tile_size, (1, 1), tiles_wide)
      pixelf = ti.cast(pixel, dtype) + 0.5

      # Initialize accumulators for all pixels in tile
      in_bounds = pixel.y < camera_height and pixel.x < camera_width

      accum_features = feature_vec(0.0)
      total_weight = dtype(0.0) if in_bounds else dtype(1.0)
      saturated = False

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = tiling.round_up(tile_point_count, tile_area)

      # Open shared memory arrays
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)
      tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)

      tile_visibility = (ti.simt.block.SharedArray((tile_area, ), dtype=vec1) 
                         if ti.static(config.compute_visibility) else None)

      for point_group_id in range(num_point_groups):
        if ti.simt.block.sync_all_nonzero(ti.i32(saturated)):
          break

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * tile_area
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]
          tile_point_id[tile_idx] = point_idx

          if ti.static(config.compute_visibility):
            tile_visibility[tile_idx] = vec1(0.0)

        ti.simt.block.sync()

        remaining_points = tile_point_count - point_group_id

        # Process all points in group for each pixel in tile
        for in_group_idx in range(min(tile_area, remaining_points)):
          if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), 
              ti.i32(saturated)):
            break

          weight = dtype(0.0)
          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

          gaussian_alpha = pdf(pixelf, mean, axis, sigma)
          alpha = point_alpha * gaussian_alpha
          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

          if alpha > ti.static(config.alpha_threshold):
            weight = alpha * (1.0 - total_weight)
            total_weight += weight

            if ti.static(config.use_alpha_blending):
              accum_features += tile_feature[in_group_idx] * weight
            else:
              # no blending - use this to compute quantile (e.g. median) along with config.saturate_threshold
              if total_weight >= ti.static(1.0 - config.saturate_threshold) and not saturated:
                accum_features = tile_feature[in_group_idx]
            
              saturated = total_weight >= ti.static(1.0 - config.saturate_threshold)

          if ti.static(config.compute_visibility):
            if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(alpha > ti.static(config.alpha_threshold))):
              # Accumulate visibility in shared memory across the warp
              weight_vec = vec1(weight)
              warp_add_vector(tile_visibility[in_group_idx], weight_vec)


        if ti.static(config.compute_visibility):
          ti.simt.block.sync()  

          if load_index < end_offset:
            point_idx = tile_point_id[tile_idx] 
            ti.atomic_add(visibility[point_idx], tile_visibility[tile_idx][0])

      # Write final results
      if in_bounds:
        image_feature[pixel.y, pixel.x] = accum_features

        if ti.static(config.use_alpha_blending):
          image_alpha[pixel.y, pixel.x] = total_weight    
        else:
          image_alpha[pixel.y, pixel.x] = dtype(total_weight > 0)

  return _forward_kernel


@cache
def forward_batch_kernel(config: RasterConfig, feature_size: int, dtype=ti.f32):
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  kernel_variant = config.kernel_variant
  if kernel_variant == "auto":
    kernel_variant = "metal_reference" if config.metal_compatible else "cuda_simt"

  pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf

  if kernel_variant != "metal_reference":
    raise ValueError(f"Unsupported batched rasterizer forward kernel_variant={kernel_variant!r}")

  @ti.kernel
  def _forward_batch_kernel_simple(
      # Input tensors
      points: ti.types.ndarray(Gaussian2D.vec, ndim=2),              # [B, N, 7] 2D gaussian parameters
      point_features: ti.types.ndarray(feature_vec, ndim=2),         # [B, N, F] gaussian features

      # Tile data structures
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [B * T] start/end range of overlapping points
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to local point index

      # Output image buffers
      image_feature: ti.types.ndarray(feature_vec, ndim=3),          # [B, H, W, F] output features
      image_alpha: ti.types.ndarray(dtype, ndim=3),                  # [B, H, W] output alpha

      # Output visibility buffer
      visibility: ti.types.ndarray(dtype, ndim=2)                    # [B, N] visibility per point
  ):
    batch_size, camera_height, camera_width = image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size
    tiles_high = (camera_height + tile_size - 1) // tile_size
    tiles_per_image = tiles_wide * tiles_high

    if ti.static(config.metal_block_dim > 0):
      ti.loop_config(block_dim=config.metal_block_dim)
    for batch_idx, y, x in image_feature:
      tile_x = x // tile_size
      tile_y = y // tile_size
      tile_id = tile_x + tile_y * tiles_wide
      batch_tile_id = batch_idx * tiles_per_image + tile_id
      start_offset, end_offset = tile_overlap_ranges[batch_tile_id]

      pixelf = ti.Vector([ti.cast(x, dtype) + 0.5, ti.cast(y, dtype) + 0.5])
      accum_features = feature_vec(0.0)
      total_weight = dtype(0.0)

      for overlap_idx in range(start_offset, end_offset):
        point_idx = overlap_to_point[overlap_idx]
        mean, axis, sigma, point_alpha = Gaussian2D.unpack(points[batch_idx, point_idx])

        gaussian_alpha = pdf(pixelf, mean, axis, sigma)
        alpha = point_alpha * gaussian_alpha
        alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

        if alpha > ti.static(config.alpha_threshold) and total_weight < ti.static(config.saturate_threshold):
          weight = alpha * (1.0 - total_weight)
          total_weight += weight

          if ti.static(config.use_alpha_blending):
            accum_features += point_features[batch_idx, point_idx] * weight
          else:
            if total_weight >= ti.static(config.median_threshold):
              accum_features = point_features[batch_idx, point_idx]

          if ti.static(config.compute_visibility):
            ti.atomic_add(visibility[batch_idx, point_idx], weight)

      image_feature[batch_idx, y, x] = accum_features
      image_alpha[batch_idx, y, x] = total_weight

  return _forward_batch_kernel_simple
