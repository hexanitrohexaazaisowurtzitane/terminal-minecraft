import numpy as np
import random
from enum import IntEnum
from noise import snoise2, snoise3
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time
import os

import numba
from numba import njit, prange
import threading
import queue

import logging
logging.basicConfig(level=logging.INFO)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# function to log to render
log_callback_func = None
def log_callback(msg):
    if log_callback_func: log_callback_func(msg)
    else: logging.info(msg)

FACE_VERTICES = np.array(
    [
        [[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]],  # +z front
        [[-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5]],  # -z back
        [[-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5]],  # +y top
        [[-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5]],  # -y bottom
        [[ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5]],  # +x right
        [[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]]   # -x left
    ], dtype=np.float32
)
NEIGHBORS = np.array(
    [
        [ 0,  0,  1],   # +z front
        [ 0,  0, -1],   # -z back
        [ 0,  1,  0],   # +y top
        [ 0, -1,  0],   # -y bottom
        [ 1,  0,  0],   # +x right
        [-1,  0,  0]    # -x left
    ], dtype=np.int8
)



@njit(parallel=True)
def compute_face_visibility(blocks, size_x, size_y, size_z):
    """precompute face visibility for all blocks in chunk"""
    # result array: blocks x 6 faces x visibility   (bool)
    visibility = np.zeros((size_x, size_y, size_z, 6), dtype=np.bool_)
    
    for x in prange(size_x):
        for y in range(size_y):
            for z in range(size_z):
                if blocks[x, y, z] == 0: continue
                is_water = blocks[x, y, z] == 9
                    
                for face_idx in range(6):
                    nx = x + NEIGHBORS[face_idx, 0]
                    ny = y + NEIGHBORS[face_idx, 1]
                    nz = z + NEIGHBORS[face_idx, 2]
                    
                    # is at chunk border
                    at_border = (
                        nx < 0 or nx >= size_x or 
                        ny < 0 or ny >= size_y or 
                        nz < 0 or nz >= size_z
                    )
                    
                    if at_border:
                        # make border faces visible if not water
                        visibility[x, y, z, face_idx] = not is_water
                    else:
                        neighbor_block = blocks[nx, ny, nz]
                        # face visible if neighbor is air or 
                        # if this is not water and neighbor is water (underwater visibility)
                        visibility[x, y, z, face_idx] = (
                            neighbor_block == 0 or 
                            (not is_water and neighbor_block == 9)
                        )
                        
    return visibility

@njit
def get_face_color(block_type, face_idx, color_lookup):
    face_type = 0
    if face_idx in (0, 1, 4, 5):  face_type = 1  # side faces
    elif face_idx == 3:           face_type = 2  # bottom face
    
    return color_lookup[block_type, face_type]

@njit
def count_visible_faces(blocks, visibility, size_x, size_y, size_z):
    """total number of visible faces to preallocate arrays."""
    count = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                if blocks[x, y, z] == 0:  continue
                for face_idx in range(6):
                    if visibility[x, y, z, face_idx]:
                        count += 1
    return count



@njit
def generate_mesh_data(blocks, color_lookup, pos_x, pos_z, size_x, size_y, size_z):
    """generate mesh data for a chunk using precomputed visibility."""
    visibility = compute_face_visibility(blocks, size_x, size_y, size_z)
    
    total_faces = count_visible_faces(blocks, visibility, size_x, size_y, size_z)
    
    # prealloc arrays : 4 vertices / face ;  3 coords / vertex
    vertices = np.zeros((total_faces * 4, 3), dtype=np.float32)
    colors = np.zeros((total_faces * 4, 3), dtype=np.float32)
    
    
    chunk_pos = np.array([pos_x * size_x, 0, pos_z * size_z], dtype=np.float32) # offset for world coords
    
    # generate mesh
    face_count = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                block_type = blocks[x, y, z]
                if block_type == 0:  continue
                
                block_pos = np.array([x, y, z], dtype=np.float32) + chunk_pos
                
                for face_idx in range(6):
                    if visibility[x, y, z, face_idx]:
                        face_color = get_face_color(block_type, face_idx, color_lookup)
                        
                        # 4 vertices for face
                        vertex_idx = face_count * 4
                        for i in range(4):
                            vertices[vertex_idx + i] = FACE_VERTICES[face_idx, i] + block_pos
                            colors[vertex_idx + i]   = face_color
                        
                        face_count += 1
    
    # flatten arrays for rendering
    return vertices[:face_count*4].reshape(-1), colors[:face_count*4].reshape(-1)

class BlockType(IntEnum):
    AIR, GRASS, DIRT, STONE, LOG, WOOD, LEAVES, SAND, CACTUS, WATER = range(10)

class BiomeType(IntEnum):
    PLAINS, MOUNTAINS, FOREST, DESERT = range(4)

# -- old --

class Chunk:
    def __init__(self, pos_x=0, pos_z=0, size_x=16, size_y=64, size_z=16, seed=0):
        self.pos_x,  self.pos_z = pos_x, pos_z
        self.size_x, self.size_y, self.size_z = size_x, size_y, size_z
        
        self.blocks = np.zeros((size_x, size_y, size_z), dtype=np.uint8)
        self.seed = seed
        
        self.block_colors = {
            BlockType.GRASS:  np.array([[0.0, 0.8, 0.0], [0.6, 0.4, 0.2], [0.0, 0.8, 0.0]], dtype=np.float32),
            BlockType.DIRT:   np.array([[0.6, 0.4, 0.2], [0.6, 0.4, 0.2], [0.5, 0.3, 0.1]], dtype=np.float32),
            BlockType.STONE:  np.array([[0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], dtype=np.float32),
            BlockType.LOG:    np.array([[0.6, 0.4, 0.2], [0.5, 0.3, 0.1], [0.6, 0.4, 0.2]], dtype=np.float32),
            BlockType.WOOD:   np.array([[0.7, 0.5, 0.3], [0.6, 0.4, 0.2], [0.7, 0.5, 0.3]], dtype=np.float32),
            BlockType.LEAVES: np.array([[0.0, 0.6, 0.0], [0.0, 0.5, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32),
            BlockType.SAND:   np.array([[0.9, 0.8, 0.6], [0.85, 0.75, 0.55], [0.9, 0.8, 0.6]], dtype=np.float32),
            BlockType.CACTUS: np.array([[0.0, 0.7, 0.0], [0.0, 0.6, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
            BlockType.WATER:  np.array([[0.0, 0.5, 1.0], [0.0, 0.45, 0.9], [0.0, 0.5, 1.0]], dtype=np.float32),
        }
        
        # color lookups
        self.color_lookup = np.zeros((len(BlockType), 3, 3), dtype=np.float32)
        for block_type, colors in self.block_colors.items():
            self.color_lookup[block_type] = colors
        
        random.seed(seed)
        self._generate_terrain()
        self.mesh_data = None
    
    


    def _get_noise(self, x, z, scale):
        world_x = x + self.pos_x * self.size_x
        world_z = z + self.pos_z * self.size_z
        return snoise2(world_x / scale, world_z / scale)

    def _get_3d_noise(self, x, y, z, scale):
        world_x = x + self.pos_x * self.size_x
        world_z = z + self.pos_z * self.size_z
        return snoise3(world_x / scale, y / scale, world_z / scale)

    def _get_biome(self, x, z):
        temperature = self._get_noise(x, z, 100)
        moisture    = self._get_noise(x, z, 80)
        
        # mountains broken, TODO
        if temperature > 0.4: return BiomeType.DESERT
        elif moisture > 0.4:  return BiomeType.FOREST
        else:                 return BiomeType.PLAINS




    def _place_tree(self, x, y, z):
        if (
            x < 2 or x >= self.size_x - 2 or 
            z < 2 or z >= self.size_z - 2 or 
            y + 5 >= self.size_y
        ): return
        
        trunk_height = random.randint(3, 5)
        for h in range(trunk_height):
            if y + h < self.size_y:
                self.blocks[x, y + h, z] = BlockType.LOG
        
        leaf_y_start = y + trunk_height - 2
        leaf_y_end   = min(y + trunk_height + 2, self.size_y)
        
        for ly in range(leaf_y_start, leaf_y_end):
            leaf_radius = 2 if ly < leaf_y_end - 1 else 1
            for lx in range(x - leaf_radius, x + leaf_radius + 1):
                for lz in range(z - leaf_radius, z + leaf_radius + 1):
                    if 0 <= lx < self.size_x and 0 <= ly < self.size_y and 0 <= lz < self.size_z:
                        if self.blocks[lx, ly, lz] == BlockType.AIR:
                            self.blocks[lx, ly, lz] = BlockType.LEAVES

    def _place_cactus(self, x, y, z):
        if (
            x < 1 or x >= self.size_x - 1 or 
            z < 1 or z >= self.size_z - 1
        ): return
            
        height = random.randint(2, 4)
        for h in range(height):
            if y + h < self.size_y:
                self.blocks[x, y + h, z] = BlockType.CACTUS



    def _generate_terrain(self):
        water_level = 27
        
        # start_time = time.time()
        # precompute and vectorize height map and biome generation
        heightmap = np.zeros((self.size_x, self.size_z), dtype=int)
        biome_map = np.zeros((self.size_x, self.size_z), dtype=int)
        
        for x in range(self.size_x):
            for z in range(self.size_z):
                base_height = self._get_noise(x, z, 50)
                biome = self._get_biome(x, z)
                biome_map[x, z] = biome
                
                if biome == BiomeType.MOUNTAINS:
                    mountain_height = self._get_noise(x, z, 30) * 20
                    height = 32 + base_height * 10 + mountain_height
                else:  height = 32 + base_height * 8
                
                heightmap[x, z] = int(min(max(height, 1), self.size_y - 1))
        
        # chunk cave mask 
        cave_mask = np.zeros((self.size_x, self.size_y, self.size_z), dtype=bool)
        for x in range(self.size_x):
            for y in range(min(40, self.size_y)):  # only below y=40
                for z in range(self.size_z):
                    cave_val = self._get_3d_noise(x, y, z, 20)
                    cave_mask[x, y, z] = cave_val > 0.7

        # fill terrain
        for x in range(self.size_x):
            for z in range(self.size_z):
                height = heightmap[x, z]
                biome  = biome_map[x, z]
                
                # water
                if height < water_level:
                    # stone floor
                    for y in range(height):
                        if not cave_mask[x, y, z]:
                            self.blocks[x, y+1, z] = BlockType.STONE
                    
                    # sand shore
                    if height == water_level - 1:
                        self.blocks[x, height, z] = BlockType.SAND
                    
                    # water above
                    for y in range(height + 1, water_level):
                        self.blocks[x, y, z] = BlockType.WATER
                    

                    continue
                
                # ground
                for y in range(height + 1):
                    if cave_mask[x, y, z]: continue
                        
                    if y == height:
                        if biome == BiomeType.DESERT:
                            self.blocks[x, y, z] = BlockType.SAND
                        else:
                            # sand near water level for beaches
                            if height <= water_level:
                                  self.blocks[x, y, z] = BlockType.SAND
                            else: self.blocks[x, y, z] = BlockType.GRASS

                    elif y > height - 4:
                        if biome == BiomeType.DESERT:
                              self.blocks[x, y, z] = BlockType.SAND
                        else: self.blocks[x, y, z] = BlockType.DIRT
                    
                    else:  self.blocks[x, y, z] = BlockType.STONE
        
        # features
        for x in range(self.size_x):
            for z in range(self.size_z):
                height = heightmap[x, z]
                biome  = biome_map[x, z]
                

                if height >= water_level and height < self.size_y - 6:
                    if   biome == BiomeType.FOREST and random.random() < 0.1:
                        self._place_tree(x, height + 1, z)
                    elif biome == BiomeType.PLAINS and random.random() < 0.02:
                        self._place_tree(x, height + 1, z)
                    elif biome == BiomeType.DESERT and random.random() < 0.02:
                        self._place_cactus(x, height + 1, z)



    def get_face_color(self, block_type, face_idx):
        if block_type == BlockType.AIR: return [0, 0, 0]
        
        face_type = 0
        if face_idx in [0, 1, 4, 5]: face_type = 1
        elif face_idx == 3:          face_type = 2
            
        return self.block_colors[block_type][face_type]

    """
    def generate_mesh_data(self):
        # replacement for Chunk.generate_mesh_data using Numba acceleration.
        if self.mesh_data is not None:
            return self.mesh_data

        #start_time = time.time()
        
        # Call the optimized mesh generation
        vertices, colors = _generate_mesh_data(
            self.blocks, self.color_lookup, 
            self.pos_x, self.pos_z, 
            self.size_x, self.size_y, self.size_z
        )
        
        self.mesh_data = (vertices, colors)
        #print(f"Mesh data generated in {time.time() - start_time:.2f} seconds")
        return self.mesh_data
    """




def generate_chunk_process(args):
    pos_x, pos_z, chunk_size, seed = args
    chunk = Chunk(pos_x, pos_z, chunk_size, 64, chunk_size, seed)
    return (pos_x, pos_z), (chunk.blocks, chunk.block_colors, chunk.color_lookup)

def process_chunk_mesh(args):
    blocks, block_colors, color_lookup, pos_x, pos_z, chunk_size = args
    
    # force color lookup into numpy array
    if not isinstance(color_lookup, np.ndarray):
        # create color lookup as numpy array
        max_block_type = max(block_colors.keys()) + 1
        color_lookup_array = np.zeros((max_block_type, 3, 3), dtype=np.float32)
        
        for block_type, colors in block_colors.items():
            color_lookup_array[block_type] = colors

        color_lookup = color_lookup_array
    
    # generate mesh data
    vertices, colors = generate_mesh_data(
        blocks, color_lookup, pos_x, pos_z, chunk_size, 64, chunk_size
    )
    
    return (pos_x, pos_z), (vertices, colors)







class ChunkManager:
    def __init__(self, chunk_size=16, render_distance=5, seed=None, max_workers=None):
        self.chunk_size      = chunk_size
        self.spawn_distance  =  render_distance
        self.render_distance = render_distance
        self.seed = seed if seed is not None else random.randint(0, 999999)
        self.chunks       = {}
        self.loaded_chunks = set()
        self.chunk_data   = {}
        self.chunk_meshes = {}
        self.mesh_generation_queue = set()
        self.chunk_load_time = 0
        self.mesh_gen_time   = 0
        self.stats = {
            "active_chunks":   0,
            #"vertices":        0,
            #"pending_loads":   0,
            #"pending_meshes":  0,
            #"last_load_time":  0,
            "total_generated": 0
        }
        
        # auto workers based on available cores
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() - 1))
        self.max_workers = max_workers
        
        # executor for async chunk generation
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.pending_futures = {}
        
        # prealloc buffers for meshes
        self.vertices = np.array([], dtype=np.float32)
        self.colors = np.array([], dtype=np.float32)
        
        
        self.distance_cache = {}
        self.pregenerate_chunks()
        
        # Chunk generation and management configs
        """
        Advanced chunk management configs,
        If you have a medium-high end hardware, adjust the following settings:
          NAME                     LOW END     MEDIUM END     HIGH END   DESCRIPTION
        * max_loads_per_frame      1           2               3         max number of chunks to load per frame
        * chunk_load_cooldown      0.1         0.05            0.01      time in seconds between chunk loads
        * priority_update_interval 1.2s        0.6s            0.3s      time interval for priority update

        * max_loaded_chunks        50          75              100       max number of chunks stored in memory
        * memory_threshold         200MB       300MB           400MB     memory threshold for chunk unloading
        * memory_check_interval    15.0s       8.0s            4.0s      time interval for memory check

        * memory_cache_size        40          60              100       size of mesh cache
        """

        self.max_loads_per_frame  = 1                   # 2
        self.chunk_load_cooldown  = 0.1                 # 0.05
        self.last_chunk_load_time = 0
        
        self.loading_priority_queue   = queue.PriorityQueue()
        self.last_priority_update     = time.time()
        self.priority_update_interval = 1.2             # 0.6
        
        self.chunk_lru = {} 
        self.max_loaded_chunks     = 75                 # 50
        self.chunk_memory_usage    = {}  
        self.memory_threshold      = 300 * 1024 * 1024  # 200MB
        self.last_memory_check     = time.time()
        self.memory_check_interval = 15.0               # 8.0
        
        self.mesh_cache_size = 40                       # 25
        self.mesh_lru = {} 
        self.mesh_vertex_counts = {} 
    





    def pregenerate_chunks(self):
        #return True
        logging.info("Pregenerating spawn chunks, please wait...")
        center_x, center_z = 0, 0

        start_time = time.time()
        
        
        chunk_coords = []
        for x in range(center_x - self.spawn_distance, center_x + self.spawn_distance + 1):
            for z in range(center_z - self.spawn_distance, center_z + self.spawn_distance + 1):
                if abs(x - center_x) + abs(z - center_z) <= self.spawn_distance:
                    chunk_coords.append((x, z))
        
        # process batches
        batch_size = min(self.max_workers * 2, len(chunk_coords))
        for i in range(0, len(chunk_coords), batch_size):
            batch = chunk_coords[i:i+batch_size]
            futures = []
            
            for x, z in batch:
                future = self.executor.submit(
                    generate_chunk_process, 
                    (x, z, self.chunk_size, self.seed)
                )
                futures.append((future, (x, z)))
                logging.info(f"Queued chunk generation for ({x}, {z})")
            
            # wait for complletion
            for future, (x, z) in futures:
                try:
                    _, (blocks, block_colors, color_lookup) = future.result()
                    # store as chunk data
                    chunk = Chunk(x, z, self.chunk_size, 64, self.chunk_size, self.seed)
                    chunk.blocks        = blocks
                    chunk.block_colors  = block_colors
                    chunk.color_lookup  = color_lookup
                    self.chunks[(x, z)] = chunk
                    self.chunk_data[(x, z)] = (blocks, block_colors, color_lookup)
                    self.loaded_chunks.add((x, z))
                    self.stats["total_generated"] += 1
                    
                    # generate immediate mesh
                    mesh_args = (blocks, block_colors, color_lookup, x, z, self.chunk_size)
                    (_, _), (vertices, colors) = process_chunk_mesh(mesh_args)
                    if len(vertices) > 0:  self.chunk_meshes[(x, z)] = (vertices, colors)
                    logging.info(f"Pregenerated Chunk at ({x}, {z})")
                except Exception as e:
                    logging.error(f"Failed to pregenerate chunk at ({x}, {z}): {e}")
        


        self.update_combined_mesh()
        logging.info(f"Pregenerated a total of {len(self.loaded_chunks)} chunks")
        logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
    


    
    def get_chunk_coords_for_position(self, pos):
        """convert world position to chunk coordinates"""
        x, y, z = pos
        chunk_x = int(x // self.chunk_size)
        chunk_z = int(z // self.chunk_size)
        return chunk_x, chunk_z
    
    def update_chunks_around_position(self, position):
        #start_time = time.time()
        center_chunk_x, center_chunk_z = self.get_chunk_coords_for_position(position)
        
        # mark chunks that should be visible using cached calculations where possible
        new_visible_chunks = set()
        for x in range(center_chunk_x - self.render_distance, center_chunk_x + self.render_distance + 1):
            for z in range(center_chunk_z - self.render_distance, center_chunk_z + self.render_distance + 1):
                dx, dz = abs(x - center_chunk_x), abs(z - center_chunk_z)
                
                # use Manhattan distance thing
                
                distance_key = (dx, dz)
                if distance_key in self.distance_cache: # distance cache
                    if self.distance_cache[distance_key] <= self.render_distance:
                        new_visible_chunks.add((x, z))
                else:
                    # bake and cache distance
                    distance = dx + dz
                    self.distance_cache[distance_key] = distance
                    if distance <= self.render_distance:
                        new_visible_chunks.add((x, z))
        
        # find chunks to update
        chunks_to_load   = new_visible_chunks - self.loaded_chunks
        chunks_to_unload = self.loaded_chunks - new_visible_chunks

        if not chunks_to_load and not chunks_to_unload: return False



        for chunk_coord in chunks_to_unload:
            if chunk_coord in self.loaded_chunks:
                self.loaded_chunks.remove(chunk_coord)
        
        
        
            
        # queue distance based priority chunks
        chunk_load_queue = []
        for x, z in chunks_to_load:
            dist = abs(x - center_chunk_x) + abs(z - center_chunk_z)
            chunk_load_queue.append((dist, x, z))
        
        
        chunk_load_queue.sort()  # sort by distance
        
        # queue chunks for loading (limit batch size to avoid lag spikes)
        # only load 1 chunk per frame to prevent lag spikes
        max_chunks_per_update = 1
        chunks_queued = 0
        
        for _, x, z in chunk_load_queue:
            if chunks_queued >= max_chunks_per_update: break
                
            if (x, z) not in self.chunks and (x, z) not in self.pending_futures:
                future = self.executor.submit(
                    generate_chunk_process, 
                    (x, z, self.chunk_size, self.seed)
                )
                self.pending_futures[(x, z)] = future
                chunks_queued += 1
                #self.stats["pending_loads"] += 1
            elif ( (x, z) in self.chunks and 
                   (x, z) not in self.mesh_generation_queue and 
                   (x, z) not in self.chunk_meshes ):
                # request mesh generation for generated chunk
                self.mesh_generation_queue.add((x, z))
                #self.stats["pending_meshes"] += 1
            
            self.loaded_chunks.add((x, z))
        


        # process completed 
        completed_futures = []
        for coord, future in list(self.pending_futures.items()):
            if future.done():
                try:
                    (x, z), (blocks, block_colors, color_lookup) = future.result()
                    # store as chunk data
                    chunk = Chunk(x, z, self.chunk_size, 64, self.chunk_size, self.seed)
                    chunk.blocks        = blocks
                    chunk.block_colors  = block_colors
                    chunk.color_lookup  = color_lookup
                    self.chunks[(x, z)] = chunk
                    self.chunk_data[(x, z)] = (blocks, block_colors, color_lookup)
                    self.stats["total_generated"] += 1
                    
                    
                    self.mesh_generation_queue.add((x, z))
                    #self.stats["pending_meshes"] += 1
                    #self.stats["pending_loads"] -= 1
                except Exception as e:
                    #logging.error(f"Chunk generation failed at ({x}, {z}): {e}")
                    log_callback(f"Chunk generation failed at ({x}, {z}): {e}")
                
                completed_futures.append(coord)
        



        for coord in completed_futures:
            del self.pending_futures[coord]
        
        # process mesh generation queue (limited to avoid lag)
        # only process 1 mesh p/ frame to maintain stable fps
        # update this for high end hardware
        self.process_mesh_queue(1) # 2
        
        
        self.stats["active_chunks"] = len(self.loaded_chunks)
        #self.stats["last_load_time"] = time.time() - start_time
        
        return len(chunks_to_load) > 0 or len(chunks_to_unload) > 0
    



    def process_mesh_queue(self, max_processing=1):
        if not self.mesh_generation_queue: return
        processed = 0
        
        
        coords_to_process = []
        for coord in list(self.mesh_generation_queue):
            if processed >= max_processing: break
                
            if coord in self.chunk_data and coord not in self.chunk_meshes:
                coords_to_process.append(coord)
                self.mesh_generation_queue.remove(coord)
                processed += 1
        
        if not coords_to_process: return
            
        # one mesh at a time
        with ProcessPoolExecutor(max_workers=1) as executor:
            for coord in coords_to_process:
                mesh_args = (
                    self.chunk_data[coord][0], self.chunk_data[coord][1], 
                    self.chunk_data[coord][2], coord[0], coord[1], self.chunk_size
                )
                
                try:
                    # main thread for smaller tasks
                    if self.chunk_size <= 16:
                        (x, z), (vertices, colors) = process_chunk_mesh(mesh_args)
                    else:
                        future = executor.submit(process_chunk_mesh, mesh_args)
                        (x, z), (vertices, colors) = future.result()
                        
                    if len(vertices) > 0:
                        self.chunk_meshes[(x, z)] = (vertices, colors)
                        # track mesh in lru cache
                        if hasattr(self, 'mesh_lru'):
                            self.mesh_lru[(x, z)] = time.time()
                            if hasattr(self, 'mesh_vertex_counts'):
                                self.mesh_vertex_counts[(x, z)] = len(vertices) // 3
                except Exception as e:
                    #logging.error(f"Mesh generation failed: {e}")
                    log_callback(f"Mesh generation failed: {e}")
        
        

        self.update_combined_mesh_incremental(coords_to_process)
    



    def update_combined_mesh_incremental(self, updated_coords):
        
        current_vertices = []
        current_colors   = []
        for coord in updated_coords:
            if coord in self.chunk_meshes:
                chunk_vertices, chunk_colors = self.chunk_meshes[coord]
                if len(chunk_vertices) > 0:
                    current_vertices.append(chunk_vertices)
                    current_colors.append(chunk_colors)
        
        # only reconstruct the full mesh when necessary
        if not hasattr(self, 'vertices') or len(self.vertices) == 0 or len(current_vertices) == 0:
            self.update_combined_mesh()
            return
            
        # append new chunks to existing mesh
        if current_vertices:
            self.vertices = np.append(self.vertices, np.concatenate(current_vertices))
            self.colors   = np.append(self.colors,   np.concatenate(current_colors))
    
    def update_combined_mesh(self):
        vertices_list = []
        colors_list   = []
        visible_chunks = [coord for coord in self.loaded_chunks if coord in self.chunk_meshes]
        
        for coord in visible_chunks:
            chunk_vertices, chunk_colors = self.chunk_meshes[coord]
            if len(chunk_vertices) > 0:
                vertices_list.append(chunk_vertices)
                colors_list.append(chunk_colors)
        
        if vertices_list:
            self.vertices = np.concatenate(vertices_list)
            self.colors   = np.concatenate(colors_list)
        else:
            self.vertices = np.array([], dtype=np.float32)
            self.colors   = np.array([], dtype=np.float32)
    


    

    def update_chunk_mesh_fast(self, chunk_coord):
        if chunk_coord not in self.chunks: return
        chunk = self.chunks[chunk_coord]
        
        # regen mesh data
        vertices, colors = generate_mesh_data(
            chunk.blocks, chunk.color_lookup,
            chunk_coord[0], chunk_coord[1],
            self.chunk_size, 64, self.chunk_size
        )
        
        # update cache
        if len(vertices) > 0:
            self.chunk_meshes[chunk_coord] = (vertices, colors)
            # track mesh in lru cache
            if hasattr(self, 'mesh_lru'):
                self.mesh_lru[chunk_coord] = time.time()
                if hasattr(self, 'mesh_vertex_counts'):
                    self.mesh_vertex_counts[chunk_coord] = len(vertices) // 3

        elif chunk_coord in self.chunk_meshes:
            del self.chunk_meshes[chunk_coord]
            if hasattr(self, 'mesh_lru') and chunk_coord in self.mesh_lru:
                del self.mesh_lru[chunk_coord]
            if hasattr(self, 'mesh_vertex_counts') and chunk_coord in self.mesh_vertex_counts:
                del self.mesh_vertex_counts[chunk_coord]
        
        
        self.update_combined_mesh()

    def cleanup(self): self.executor.shutdown(wait=False)    
  



class ChunkThread(threading.Thread):
    """Dedicated thread for chunk loading and mesh generation"""
    def __init__(self, chunk_manager):
        super().__init__(daemon=True)
        self.running = True
        self.chunk_manager    = chunk_manager
        self.generation_queue = queue.PriorityQueue()
        self.mesh_queue       = queue.PriorityQueue()
        self.completed_chunks = queue.Queue()
        self.completed_meshes = queue.Queue()
        # lock for safe thread access
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            # generate chujunks
            try:
                priority, task_type, args = self.generation_queue.get(block=True, timeout=0.1)
                
                if task_type == "generate":
                    x, z = args[0], args[1]
                    try:
                        result = generate_chunk_process(args)
                        self.completed_chunks.put(result)
                    except Exception as e:
                        #logging.error(f"Thread chunk generation error at ({x}, {z}): {e}")
                        log_callback(f"Thread chunk generation error at ({x}, {z}): {e}")
                    
                    
                    self.generation_queue.task_done()
                    
            except queue.Empty: pass
                
            # generate meshes
            try:
                priority, task_id, args = self.mesh_queue.get(block=False)
                
                try:
                    result = process_chunk_mesh(args)
                    self.completed_meshes.put(result)
                except Exception as e:
                    #logging.error(f"Thread mesh generation error: {e}")
                    log_callback(f"Thread mesh generation error: {e}")
                self.mesh_queue.task_done()
                
            except queue.Empty: time.sleep(0.01) # pass
    
    def queue_chunk_generation(self, args, priority=0):
        # lower = higher priority
        self.generation_queue.put((priority, "generate", args))
    
    def queue_mesh_generation(self, args, priority=0):
        # lower = higher priority
        # task id to avoid comparing numpy arrays
        task_id = id(args)
        self.mesh_queue.put((priority, task_id, args))
    
    def stop(self):
        # thread stop
        self.running = False
        self.join(timeout=1.0)


class ThreadedChunkManager(ChunkManager):
    def __init__(self, chunk_size=16, render_distance=5, seed=None, max_workers=None):
        
        logging.info(f"Chunk Manager Initialized with Render_Distance={render_distance}, Max_Workers={max_workers}, seed={seed}")
        
        # CHANGE THESE ONES if using threaded chunk generation (yes, you are)
        # Chunk generation and management configs
        """
        Advanced chunk management configs,
        If you have a medium-high end hardware, adjust the following settings:
          NAME                     LOW END     MEDIUM END     HIGH END   DESCRIPTION
        * max_unloads_per_frame    1           3               5         max number of chunks to unload per frame
        * position_history_max     3           5               10        max number of positions to store in history
        
        * max_loads_per_frame      1           3               5         max number of chunks to load per frame
        * chunk_load_cooldown      0.1         0.05            0.01      time in seconds between chunk loads
        * priority_update_interval 1.2s        0.6s            0.3s      time interval for priority update

        * max_loaded_chunks        50          75              100       max number of chunks stored in memory
        * memory_threshold         200MB       300MB           400MB     memory threshold for chunk unloading
        * memory_check_interval    15.0s       8.0s            4.0s      time interval for memory check

        * mesh_caceh_size          25          40              100       max number of meshes stored in cache
        * mesh_rebuild_cooldown    1.2         0.6             0.1       time in seconds between mesh rebuilds

        there are other small-impact configs below, useless to change tbh
        """


                                                            # old vars
        self.unload_queue = queue.PriorityQueue()
        self.chunks_to_remove_from_mesh = set()
        self.max_unloads_per_frame      = 3                 # 1
        
        
        self.position_history = np.zeros((3, 3), dtype=np.float32)  # last 3
        self.position_history_index     = 0
        self.position_history_max       = 3                 # 5
        

        self.max_loads_per_frame        = 1                 # 2
        self.chunk_load_cooldown        = 0.1               # 0.05
        self.last_chunk_load_time       = 0
        
        self.chunk_lru                 = {} 
        self.max_loaded_chunks         = 75                 # 50
        self.chunk_memory_usage        = {}
        self.memory_threshold          = 300 * 1024 * 1024  # 300MB
        self.last_memory_check         = time.time()
        self.memory_check_interval     = 15.0               # 8.0
        
        self.mesh_cache_size           = 40                 # 25
        self.mesh_lru                  = {} 
        self.mesh_vertex_counts        = {}  
        
        self.loading_priority_queue    = queue.PriorityQueue()
        self.last_priority_update      = time.time()
        self.priority_update_interval  = 1.2               # 0.6

        
        _WIDTH, _HEIGHT = os.get_terminal_size()
        logging.info(f"Max_Unloads={self.max_unloads_per_frame}, Max_History={self.position_history_max}, Max_Loads={self.max_loads_per_frame}, Load_Cooldown={self.chunk_load_cooldown}, Max_Loaded={self.max_loaded_chunks}, Screen_Size={_WIDTH}x{_HEIGHT}")
        logging.info(f"Memory_Threshold={self.memory_threshold}MB, Check_Intreval={self.memory_check_interval}, Cache_Size={self.mesh_cache_size}, Priority_Interval={self.priority_update_interval}")
        print("-"*_WIDTH)
        
        

        super().__init__(chunk_size, render_distance, seed, max_workers)
        
        # replace executor with thread
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        self.chunk_thread = ChunkThread(self)
        self.chunk_thread.start()
        
        
        self.chunks_being_generated = set()
        self.meshes_being_generated = set()
        
        
        self.high_priority_zone = 2
        
        self.mesh_buffer_dirty = False
        self.last_mesh_rebuild_time = 0
        self.mesh_rebuild_cooldown = 0.8  # 0.3

    def set_log_callback(self, render_log_function):
        global log_callback_func
        log_callback_func = render_log_function
        print("INFO: chunk callback assigned")

    def _update_loading_priorities(self, camera_pos):
        """update loading priorities based on camera position"""
        current_time = time.time()
        if current_time - self.last_priority_update < self.priority_update_interval: return
        
        self.last_priority_update = current_time
        
        while not self.loading_priority_queue.empty(): # clear queue
            self.loading_priority_queue.get()
            
        
        center_chunk_x, center_chunk_z = self.get_chunk_coords_for_position(camera_pos)
        
        # bake priorities for all chunks in render distance
        for x in range(center_chunk_x - self.render_distance, center_chunk_x + self.render_distance + 1):
            for z in range(center_chunk_z - self.render_distance, center_chunk_z + self.render_distance + 1):
                dx, dz = abs(x - center_chunk_x), abs(z - center_chunk_z)
                distance = dx + dz
                
                if distance <= self.render_distance:
                    priority = 1000 - distance * 10
                    self.loading_priority_queue.put((-priority, (x, z)))



    def _manage_memory(self):
        """manage memory usage by unloading less used chunks"""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval: return
        self.last_memory_check = current_time
        
        
        total_memory = sum(self.chunk_memory_usage.values())
        
        if total_memory > self.memory_threshold:
            # sort by last access time
            sorted_chunks = sorted(self.chunk_lru.items(), key=lambda x: x[1])
            
            for chunk_coord, _ in sorted_chunks:
                if total_memory <= self.memory_threshold * 0.8:  # leave 20% buffer
                    break
                    
                if chunk_coord in self.chunk_memory_usage:
                    total_memory -= self.chunk_memory_usage[chunk_coord]
                    self.unload_chunk(chunk_coord)

    def _manage_mesh_cache(self):
        """manage mesh cache by removing less used meshes"""
        if len(self.chunk_meshes) <= self.mesh_cache_size: return
            
        # sort cached meshes by last access time
        available_meshes = [(coord, time) for coord, time in self.mesh_lru.items() if coord in self.chunk_meshes]
        if not available_meshes: return 
        
        
        # remove excess
        sorted_meshes = sorted(available_meshes, key=lambda x: x[1])
        while len(self.chunk_meshes) > self.mesh_cache_size and sorted_meshes:
            chunk_coord, _ = sorted_meshes.pop(0)
            if chunk_coord in self.chunk_meshes:
                del self.chunk_meshes[chunk_coord]
                if chunk_coord in self.mesh_lru:
                    del self.mesh_lru[chunk_coord]
                if chunk_coord in self.mesh_vertex_counts:
                    del self.mesh_vertex_counts[chunk_coord]

    def update_chunks_around_position(self, position):
        self._update_loading_priorities(position)
        self._manage_memory()
        self._manage_mesh_cache()
        self.process_unload_queue()

        current_time = time.time()
        
        # updaye position history for movement prediction
        if self.position_history_index == 0 and np.all(self.position_history == 0):
            self.position_history[self.position_history_index] = np.copy(position)
            self.position_history_index = (self.position_history_index + 1) % self.position_history_max

        elif np.linalg.norm(position - self.position_history[self.position_history_index - 1]) > 0.5:
            # update history with circular buffer
            self.position_history[self.position_history_index] = np.copy(position)
            self.position_history_index = (self.position_history_index + 1) % self.position_history_max
        
        
        center_chunk_x, center_chunk_z = self.get_chunk_coords_for_position(position)
        new_visible_chunks = set()

        for x in range(center_chunk_x - self.render_distance, center_chunk_x + self.render_distance + 1):
            for z in range(center_chunk_z - self.render_distance, center_chunk_z + self.render_distance + 1):
                dx, dz = abs(x - center_chunk_x), abs(z - center_chunk_z)
                distance_key = (dx, dz)
                
                if distance_key in self.distance_cache:
                    if self.distance_cache[distance_key] <= self.render_distance:
                        new_visible_chunks.add((x, z))
                else:
                    distance = dx + dz
                    self.distance_cache[distance_key] = distance
                    if distance <= self.render_distance:
                        new_visible_chunks.add((x, z))
        


        chunks_to_load   = new_visible_chunks - self.loaded_chunks
        chunks_to_unload = self.loaded_chunks - new_visible_chunks
        
        # queue unloadings with distance based priority
        for chunk_coord in chunks_to_unload:
            if chunk_coord in self.loaded_chunks:
                dx = abs(chunk_coord[0] - center_chunk_x)
                dz = abs(chunk_coord[1] - center_chunk_z)
                priority = 1000 - (dx + dz)
                self.unload_queue.put((priority, chunk_coord))
        
        # load chunks based on priority
        chunks_loaded = 0
        while not self.loading_priority_queue.empty() and chunks_loaded < self.max_loads_per_frame:
            try:
                _, (x, z) = self.loading_priority_queue.get_nowait()
                if (
                    (x, z) in chunks_to_load and 
                    (x, z) not in self.chunks and 
                    (x, z) not in self.chunks_being_generated
                    ):
                    self.chunk_thread.queue_chunk_generation(
                        (x, z, self.chunk_size, self.seed),
                        priority=1000  # highest priority for visible chunks
                    )
                    self.chunks_being_generated.add((x, z))
                    self.loaded_chunks.add((x, z))
                    self.chunk_lru[(x, z)] = current_time
                    chunks_loaded += 1
            except queue.Empty:  break
        
        
        self.process_completed_chunks()
        self.process_completed_meshes()
        self.stats["active_chunks"] = len(self.loaded_chunks)
        
        return len(chunks_to_load) > 0 or len(chunks_to_unload) > 0

    def unload_chunk(self, chunk_coord):
        if chunk_coord in self.chunks:
            # track memory usage before unloading
            if chunk_coord in self.chunk_memory_usage:
                del self.chunk_memory_usage[chunk_coord]
            


            del self.chunks[chunk_coord]
            if chunk_coord in self.chunk_data:
                del self.chunk_data[chunk_coord]
            if chunk_coord in self.chunk_meshes:
                del self.chunk_meshes[chunk_coord]
            if chunk_coord in self.loaded_chunks:
                self.loaded_chunks.remove(chunk_coord)
            if chunk_coord in self.chunk_lru:
                del self.chunk_lru[chunk_coord]
            
            # remove from queues
            if chunk_coord in self.chunks_being_generated:
                self.chunks_being_generated.remove(chunk_coord)
            if chunk_coord in self.meshes_being_generated:
                self.meshes_being_generated.remove(chunk_coord)
            
            

            self.mesh_buffer_dirty = True



    def process_unload_queue(self):
        unloads_this_frame = 0
        unloaded_chunks   = []
        
        while not self.unload_queue.empty() and unloads_this_frame < self.max_unloads_per_frame:
            try:
                _, chunk_coord = self.unload_queue.get_nowait()
                if chunk_coord in self.loaded_chunks:
                    
                    if chunk_coord in self.chunk_meshes:
                        self.chunks_to_remove_from_mesh.add(chunk_coord)
                    
                    
                    self.loaded_chunks.remove(chunk_coord)
                    if chunk_coord in self.chunks:
                        del self.chunks[chunk_coord]
                    if chunk_coord in self.chunk_data:
                        del self.chunk_data[chunk_coord]
                    if chunk_coord in self.chunk_lru:
                        del self.chunk_lru[chunk_coord]
                    
                    # track for batch removal from mesh
                    unloaded_chunks.append(chunk_coord)
                    unloads_this_frame += 1
                    
                    self.mesh_buffer_dirty = True
                
                self.unload_queue.task_done()
            except queue.Empty:  break
        
        # schedule mesh update bcus unloading
        if unloaded_chunks and self.chunks_to_remove_from_mesh: return True
        return False

    def update_mesh_after_unload(self):
        if not self.chunks_to_remove_from_mesh: return
        current_time = time.time()
        
        if current_time - self.last_mesh_rebuild_time < self.mesh_rebuild_cooldown:
            return
            
        self.last_mesh_rebuild_time = current_time
        
        # remove mesh for unloaded chunks
        for coord in list(self.chunks_to_remove_from_mesh):
            if coord in self.chunk_meshes:
                del self.chunk_meshes[coord]
        


        # force mesh rebuild
        if hasattr(self, 'vertices'):
            self.vertices = np.array([], dtype=np.float32)
            self.colors = np.array([], dtype=np.float32)
        
        # track added chunks
        processed_chunks = set()
        for coord in self.loaded_chunks:
            if coord in self.chunk_meshes and coord not in self.chunks_to_remove_from_mesh and coord not in processed_chunks:
                vertices, colors = self.chunk_meshes[coord]
                if len(vertices) > 0:
                    
                    if len(self.vertices) == 0:
                        self.vertices = vertices
                        self.colors = colors
                    else:
                        self.vertices = np.append(self.vertices, vertices)
                        self.colors = np.append(self.colors, colors)
                processed_chunks.add(coord)

        self.chunks_to_remove_from_mesh.clear()
        self.mesh_buffer_dirty = False
    

    def process_completed_chunks(self):
        completed_count = 0
        max_completions_per_frame = 2  # 3
        
        
        while not self.chunk_thread.completed_chunks.empty() and completed_count < max_completions_per_frame:
            try:
                (x, z), (blocks, block_colors, color_lookup) = self.chunk_thread.completed_chunks.get_nowait()
                
                
                chunk = Chunk(x, z, self.chunk_size, 64, self.chunk_size, self.seed)
                chunk.blocks = blocks
                chunk.block_colors = block_colors
                chunk.color_lookup = color_lookup
                self.chunks[(x, z)] = chunk
                self.chunk_data[(x, z)] = (blocks, block_colors, color_lookup)
                self.stats["total_generated"] += 1
                
                # remove from in progress set
                if (x, z) in self.chunks_being_generated:
                    self.chunks_being_generated.remove((x, z))
                
                # queue mesh with low priority
                if (x, z) not in self.meshes_being_generated:
                    mesh_args = (
                        blocks, block_colors, color_lookup, x, z, self.chunk_size
                    )
                    # use distance priority only
                    priority = float(abs(x) + abs(z))
                    self.chunk_thread.queue_mesh_generation(mesh_args, priority=priority)
                    self.meshes_being_generated.add((x, z))
                
                completed_count += 1
                self.chunk_thread.completed_chunks.task_done()
                
            except queue.Empty:  break
            except Exception as e:
                log_callback(f"Error processing completed chunk: {e}")
                #logging.error(f"Error processing completed chunk: {e}")
                self.chunk_thread.completed_chunks.task_done()
    
    
    def process_completed_meshes(self):
        updated_coords      = []
        max_meshes_per_frame = 2  # 5
        processed_count      = 0
        


        while not self.chunk_thread.completed_meshes.empty() and processed_count < max_meshes_per_frame:
            try:
                (x, z), (vertices, colors) = self.chunk_thread.completed_meshes.get_nowait()
                
                if len(vertices) > 0:
                    self.chunk_meshes[(x, z)] = (vertices, colors)
                    # track mesh in cache
                    self.mesh_lru[(x, z)] = time.time()
                    self.mesh_vertex_counts[(x, z)] = len(vertices) // 3
                
                # remove from in progress set
                if (x, z) in self.meshes_being_generated:
                    self.meshes_being_generated.remove((x, z))
                
                updated_coords.append((x, z))
                processed_count += 1
                self.chunk_thread.completed_meshes.task_done()
                
            except queue.Empty: break
            except Exception as e:
                #logging.error(f"Error processing completed mesh: {e}")
                log_callback(f"Error processing completed mesh: {e}")
                self.chunk_thread.completed_meshes.task_done()
        
        # update if any new chunks are processed
        if updated_coords and len(updated_coords) > 0:
            self.update_combined_mesh_incremental(updated_coords)

    def process_mesh_queue(self, max_processing=1):
        # this is just a stub, actual processing happens in the worker thread
        # well use this method to check for completed meshes
        self.process_completed_meshes()
    
    def cleanup(self):
        if hasattr(self, 'chunk_thread'): self.chunk_thread.stop()
        if hasattr(self, 'executor'): self.executor.shutdown(wait=False)

      
        
        
        
        
        
        
        
        
