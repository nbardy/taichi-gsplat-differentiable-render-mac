from functools import cache
import math
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2
import torch


from taichi_splatting import backend_sort
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.taichi_lib.f32 import (Gaussian2D)
from taichi_splatting.taichi_lib.conversions import torch_taichi

from taichi_splatting.taichi_lib.grid_query import make_grid_query
from taichi_splatting.taichi_queue import queued

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)


@cache
def tile_mapper(config:RasterConfig, use_depth16=False, sort_backend:backend_sort.SortBackend="auto"):

  if use_depth16 is False:
    max_tile = 65535
    key_type = torch.uint64
    end_sort_bit = 48

    @ti.func
    def make_sort_key(depth:ti.f32, tile_id:ti.i32):
        assert depth >= 0, f"depth {depth} cannot be negative for int 32 key!"

        # non negative float reinterpreted as int retains the same order
        # high bits store the tile ID (most significant)
        depth_key = ti.bit_cast(depth, ti.u32)
        return  ti.cast(depth_key, ti.u64) | (ti.cast(tile_id, ti.u64) << 32)
  
    @ti.func
    def get_tile_id(key:ti.u64) -> ti.i32:
      return ti.cast(key >> 32, ti.i32)


  else:
    max_tile = 65535
    key_type = torch.uint32
    end_sort_bit = 32

    @ti.func
    def make_sort_key(depth:ti.f32, tile_id:ti.i32):
        
        # float quantized to 16 bits, then cast to 16 bit int
        # high bits store the tile ID (most significant)

        return (ti.cast(ti.math.clamp(depth, 0, 1) * 65535, ti.u32)
            |  (ti.cast(tile_id, ti.u32) << 16))

  
    @ti.func
    def get_tile_id(key:ti.u32) -> ti.i32:
       return ti.cast(key >> 16, ti.i32)


  tile_size = config.tile_size
  grid_query = make_grid_query(
    tile_size=tile_size, 
    alpha_threshold=config.alpha_threshold)
  
  
  @ti.kernel
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),  
      image_size: ivec2,

      # outputs
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
      ti.loop_config(block_dim=128)
      for idx in range(gaussians.shape[0]):
          query = grid_query(gaussians[idx], image_size)
          counts[idx] =  query.count_tiles()





  @ti.kernel
  def find_ranges_kernel(
      sorted_keys: ti.types.ndarray(torch_taichi[key_type], ndim=1),  # (M)
      # output tile_ranges (tile id -> start, end)
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),   
  ):  
    ti.loop_config(block_dim=1024)
    for idx in range(sorted_keys.shape[0]):
        # tile id is in the 32 high bits of the 64 bit key
        tile_id = get_tile_id(sorted_keys[idx])
        
        next_tile_id = max_tile
        if idx + 1 < sorted_keys.shape[0]:
           next_tile_id = get_tile_id(sorted_keys[idx + 1])

        
        if tile_id != next_tile_id:
            tile_ranges[tile_id][1] = idx + 1

            if next_tile_id < max_tile:
              tile_ranges[next_tile_id][0] = idx + 1

  @ti.kernel
  def generate_sort_keys_kernel(
      depths: ti.types.ndarray(ti.f32, ndim=1),  # (M)
      gaussians : ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M)
      cumulative_overlap_counts: ti.types.ndarray(ti.i32, ndim=1),  # (M)
      # (K), K = sum(num_overlap_tiles)
      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(torch_taichi[key_type], ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):
    tiles_wide = image_size.x // tile_size

    ti.loop_config(block_dim=128)
    for idx in range(cumulative_overlap_counts.shape[0]):
      query = grid_query(gaussians[idx], image_size)
      key_idx = cumulative_overlap_counts[idx]
      depth = depths[idx]


      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile
          tile_id = tile.x + tile.y * tiles_wide
      
          key = make_sort_key(depth, tile_id)

          # sort based on tile_id, depth
          overlap_sort_key[key_idx] = key
          overlap_to_point[key_idx] = idx # map overlap index back to point index
          key_idx += 1

  def sort_tile_depths(depths:torch.Tensor, tile_overlap_ranges:torch.Tensor, cum_overlap_counts:torch.Tensor, total_overlap:int, image_size):

    overlap_key = torch.empty((total_overlap, ), dtype=key_type, device=cum_overlap_counts.device)
    overlap_to_point = torch.empty((total_overlap, ), dtype=torch.int32, device=cum_overlap_counts.device)

    generate_sort_keys_kernel(depths.squeeze(1).contiguous(), tile_overlap_ranges, cum_overlap_counts, image_size,
                              overlap_key, overlap_to_point)

    overlap_key, overlap_to_point  = backend_sort.radix_sort_pairs(
      overlap_key, overlap_to_point, end_bit=end_sort_bit, backend=sort_backend)
    return overlap_key, overlap_to_point
  
  def generate_tile_overlaps(gaussians, image_size):
    overlap_counts = torch.empty( (gaussians.shape[0], ), dtype=torch.int32, device=gaussians.device)

    if gaussians.shape[0] > 0:
      tile_overlaps_kernel(gaussians, ivec2(image_size), overlap_counts)

      cum_overlap_counts, total_overlap = backend_sort.full_cumsum(overlap_counts, backend=sort_backend)
      return cum_overlap_counts[:-1], total_overlap
    else:
      return torch.empty((0, ), dtype=torch.int32, device=gaussians.device), 0

  @beartype
  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):


    image_size = pad_to_tile(image_size, tile_size)
    tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)

    assert tile_shape[0] * tile_shape[1] < max_tile, \
      f"tile dimensions {tile_shape} for image size {image_size} exceed maximum tile count (16 bit id), try increasing tile_size" 


    with torch.no_grad():
      cum_overlap_counts, total_overlap = generate_tile_overlaps(
        gaussians, image_size)
            

      # This needs to be initialised to zeros (not empty)
      # as sometimes there are no overlaps for a tile
      tile_ranges = torch.zeros((*tile_shape, 2), dtype=torch.int32, device=gaussians.device)

      if total_overlap > 0:
        overlap_key, overlap_to_point = sort_tile_depths(
          depths, gaussians, cum_overlap_counts, total_overlap, image_size)
        
        find_ranges_kernel(overlap_key, tile_ranges.view(-1, 2))
      else:
        overlap_to_point = torch.empty((0, ), dtype=torch.int32, device=gaussians.device)

      return overlap_to_point, tile_ranges
      
  return queued(f)


@cache
def compact_bucket_tile_mapper(config: RasterConfig):
  tile_size = config.tile_size
  grid_query = make_grid_query(
    tile_size=tile_size,
    alpha_threshold=config.alpha_threshold)

  @ti.func
  def make_bucket_key(depth: ti.f32, point_idx: ti.i32) -> ti.i64:
    depth_key = ti.bit_cast(depth, ti.u32)
    return (ti.cast(depth_key, ti.i64) << 32) | ti.cast(point_idx, ti.i64)

  @ti.kernel
  def count_compact_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),
      image_size: ivec2,
      tile_counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
    tiles_wide = image_size.x // tile_size

    for tile_id in range(tile_counts.shape[0]):
      tile_counts[tile_id] = 0

    ti.loop_config(block_dim=128)
    for point_idx in range(gaussians.shape[0]):
      query = grid_query(gaussians[point_idx], image_size)

      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile
          tile_id = tile.x + tile.y * tiles_wide
          ti.atomic_add(tile_counts[tile_id], 1)

  @ti.kernel
  def init_compact_ranges_kernel(
      cumulative_counts: ti.types.ndarray(ti.i32, ndim=1),
      write_offsets: ti.types.ndarray(ti.i32, ndim=1),
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
  ):
    for tile_id in range(tile_ranges.shape[0]):
      start = cumulative_counts[tile_id]
      end = cumulative_counts[tile_id + 1]
      write_offsets[tile_id] = start
      tile_ranges[tile_id][0] = start
      tile_ranges[tile_id][1] = end

  @ti.kernel
  def fill_compact_tile_overlaps_kernel(
      depths: ti.types.ndarray(ti.f32, ndim=1),
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),
      image_size: ivec2,
      write_offsets: ti.types.ndarray(ti.i32, ndim=1),
      overlap_keys: ti.types.ndarray(ti.i64, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
  ):
    tiles_wide = image_size.x // tile_size

    ti.loop_config(block_dim=128)
    for point_idx in range(gaussians.shape[0]):
      query = grid_query(gaussians[point_idx], image_size)
      depth = depths[point_idx]

      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile
          tile_id = tile.x + tile.y * tiles_wide
          out_idx = ti.atomic_add(write_offsets[tile_id], 1)
          overlap_keys[out_idx] = make_bucket_key(depth, point_idx)
          overlap_to_point[out_idx] = point_idx

  @ti.kernel
  def compact_insertion_sort_kernel(
      overlap_keys: ti.types.ndarray(ti.i64, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
  ):
    ti.loop_config(block_dim=128)
    for tile_id in range(tile_ranges.shape[0]):
      start, end = tile_ranges[tile_id]

      for i in range(start + 1, end):
        key = overlap_keys[i]
        value = overlap_to_point[i]
        j = i - 1

        while j >= start and overlap_keys[j] > key:
          overlap_keys[j + 1] = overlap_keys[j]
          overlap_to_point[j + 1] = overlap_to_point[j]
          j -= 1

        overlap_keys[j + 1] = key
        overlap_to_point[j + 1] = value

  @beartype
  def f(gaussians: torch.Tensor, depths: torch.Tensor, image_size: Tuple[Integral, Integral]):
    image_size = pad_to_tile(image_size, tile_size)
    tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)
    tile_count = tile_shape[0] * tile_shape[1]

    with torch.no_grad():
      tile_counts = torch.empty((tile_count,), dtype=torch.int32, device=gaussians.device)
      count_compact_tile_overlaps_kernel(gaussians, ivec2(image_size), tile_counts)
      ti.sync()

      cumulative_counts, total_overlap = backend_sort.full_cumsum(tile_counts, backend="taichi_field")
      tile_ranges = torch.empty((tile_count, 2), dtype=torch.int32, device=gaussians.device)
      write_offsets = torch.empty((tile_count,), dtype=torch.int32, device=gaussians.device)
      init_compact_ranges_kernel(cumulative_counts, write_offsets, tile_ranges)

      if total_overlap == 0:
        return torch.empty((0,), dtype=torch.int32, device=gaussians.device), tile_ranges.view(*tile_shape, 2)

      overlap_keys = torch.empty((total_overlap,), dtype=torch.int64, device=gaussians.device)
      overlap_to_point = torch.empty((total_overlap,), dtype=torch.int32, device=gaussians.device)
      fill_compact_tile_overlaps_kernel(
        depths.squeeze(1).contiguous(),
        gaussians,
        ivec2(image_size),
        write_offsets,
        overlap_keys,
        overlap_to_point)

      compact_insertion_sort_kernel(overlap_keys, overlap_to_point, tile_ranges)

      return overlap_to_point, tile_ranges.view(*tile_shape, 2)

  return queued(f)


@cache
def ordered_tile_mapper(config: RasterConfig):
  tile_size = config.tile_size
  grid_query = make_grid_query(
    tile_size=tile_size,
    alpha_threshold=config.alpha_threshold)

  @ti.kernel
  def count_ordered_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),
      image_size: ivec2,
      counts: ti.types.ndarray(ti.i32, ndim=1),
  ):
    tiles_wide = image_size.x // tile_size

    for tile_id in range(counts.shape[0]):
      tile = ivec2(tile_id % tiles_wide, tile_id // tiles_wide)
      count = 0

      for idx in range(gaussians.shape[0]):
        query = grid_query(gaussians[idx], image_size)
        rel_tile = tile - query.min_tile
        if (rel_tile.x >= 0 and rel_tile.y >= 0
            and rel_tile.x < query.tile_span.x and rel_tile.y < query.tile_span.y
            and query.test_tile(rel_tile)):
          count += 1

      counts[tile_id] = count

  @ti.kernel
  def ranges_from_cumsum_kernel(
      cumulative_counts: ti.types.ndarray(ti.i32, ndim=1),
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
  ):
    for tile_id in range(tile_ranges.shape[0]):
      tile_ranges[tile_id][0] = cumulative_counts[tile_id]
      tile_ranges[tile_id][1] = cumulative_counts[tile_id + 1]

  @ti.kernel
  def fill_ordered_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1),
      image_size: ivec2,
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
  ):
    tiles_wide = image_size.x // tile_size

    for tile_id in range(tile_ranges.shape[0]):
      tile = ivec2(tile_id % tiles_wide, tile_id // tiles_wide)
      write_idx = tile_ranges[tile_id][0]

      for point_idx in range(gaussians.shape[0]):
        query = grid_query(gaussians[point_idx], image_size)
        rel_tile = tile - query.min_tile
        if (rel_tile.x >= 0 and rel_tile.y >= 0
            and rel_tile.x < query.tile_span.x and rel_tile.y < query.tile_span.y
            and query.test_tile(rel_tile)):
          overlap_to_point[write_idx] = point_idx
          write_idx += 1

  @beartype
  def f(gaussians: torch.Tensor, depths: torch.Tensor, image_size: Tuple[Integral, Integral]):
    # This path preserves input order within every tile. It is correct only when
    # callers have already supplied gaussians in front-to-back depth order.
    image_size = pad_to_tile(image_size, tile_size)
    tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)
    tile_count = tile_shape[0] * tile_shape[1]

    with torch.no_grad():
      counts = torch.empty((tile_count,), dtype=torch.int32, device=gaussians.device)
      count_ordered_tile_overlaps_kernel(gaussians, ivec2(image_size), counts)
      cumulative_counts, total_overlap = backend_sort.full_cumsum(counts, backend="taichi_field")

      tile_ranges = torch.empty((tile_count, 2), dtype=torch.int32, device=gaussians.device)
      ranges_from_cumsum_kernel(cumulative_counts, tile_ranges)

      overlap_to_point = torch.empty((total_overlap,), dtype=torch.int32, device=gaussians.device)
      if total_overlap > 0:
        fill_ordered_tile_overlaps_kernel(gaussians, ivec2(image_size), tile_ranges, overlap_to_point)

      return overlap_to_point, tile_ranges.view(*tile_shape, 2)

  return queued(f)


@beartype
def map_to_tiles(gaussians : torch.Tensor, depth:torch.Tensor, 
                 image_size:Tuple[Integral, Integral],
                 config:RasterConfig,
                 use_depth16:bool=False,
                 sort_backend:backend_sort.SortBackend="auto"
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 7) torch.Tensor of packed gaussians, N is the number of gaussians
     depth: (N, 1)  torch.Tensor of depths (float32)
     image_size: (2, ) tuple of ints, (width, height)
     tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (M, 2) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """

  assert gaussians.ndim == 2 and gaussians.shape[1] == Gaussian2D.vec.n, f"gaussians must be Nx{Gaussian2D.vec.n} got {gaussians.shape}"
  assert depth.ndim == 2 and depth.shape[1] == 1, f"depths must be Nx1, got {depth.shape}"

  if sort_backend == "ordered_taichi":
    mapper = ordered_tile_mapper(config)
    return mapper(gaussians, depth, image_size)
  if sort_backend == "bucket_taichi":
    mapper = compact_bucket_tile_mapper(config)
    return mapper(gaussians, depth, image_size)

  mapper = tile_mapper(config, use_depth16=use_depth16, sort_backend=sort_backend)
  return mapper(gaussians, depth, image_size)
