import curses
import collections
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pygame
import time
import math
import logging

class ColorManager:
    def __init__(self):
        self.color_cache  = {}
        self.xterm_colors = self._precompute_xterm_colors()
        

    def _precompute_xterm_colors(self, idx=0):
        # store colors as numpy array for fast comparison
        colors = np.zeros((216, 3), dtype=np.uint8)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    colors[idx] = [
                        0 if r == 0 else (40 + r * 40) if r < 5 else 255,
                        0 if g == 0 else (40 + g * 40) if g < 5 else 255,
                        0 if b == 0 else (40 + b * 40) if b < 5 else 255
                    ]
                    idx += 1
        return colors
    
    def rgb_to_color_index(self, r, g, b):
        key = (r, g, b)
        if key in self.color_cache:
            return self.color_cache[key]
        
        rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        distances = np.sum((self.xterm_colors - rgb) ** 2, axis=1)
        best_idx = np.argmin(distances) + 16
        
        self.color_cache[key] = best_idx
        return best_idx

class TerminalRenderer:
    def __init__(self, chunk_manager):
        self.stdscr = None
        pygame.init()
        
        self.frame_count, self.fps = 0, 0
        self.last_time = time.time()
        
        self.chunk_manager = chunk_manager
        
        self.camera_position = np.array([0.0, 20.0, 0.0], dtype=np.float32)
        self.camera_front    = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up       = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.movement_speed  = 0.5
        
        self.left_dragging = False
        self.last_mouse_x  = self.last_mouse_y = 0
        self.mouse_sensitivity = 0.5
        self.yaw, self.pitch   = -90.0, 0.0
        
        self.moving_forward = self.moving_backward = False
        self.moving_left    = self.moving_right    = False
        self.moving_up      = self.moving_down     = False


        self.selected_block   = 1 
        self.hotbar_blocks    = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.hotbar_width     = 9  # slots
        self.hotbar_slot_size = 8  # smaller slot size
        self.hotbar_padding   = 0  # min padding
        self.hotbar_height    = 8  # smaller height
        self.hotbar_y_offset  = 2  # bottom padding




        from chunk import BlockType
        self.block_colors = {
            BlockType.GRASS:  (0.0, 0.8, 0.0),
            BlockType.DIRT:   (0.6, 0.4, 0.2),
            BlockType.STONE:  (0.7, 0.7, 0.7),
            BlockType.LOG:    (0.6, 0.4, 0.2),
            BlockType.WOOD:   (0.7, 0.5, 0.3),
            BlockType.LEAVES: (0.0, 0.6, 0.0),
            BlockType.SAND:   (0.9, 0.8, 0.6),
            BlockType.CACTUS: (0.0, 0.7, 0.0),
            BlockType.WATER:  (0.0, 0.5, 1.0)
        }
        self.block_names = {
            1: "Grass", 2: "Dirt",   3: "Stone",
            4: "Log",   5: "Wood",   6: "Leaves",
            7: "Sand",  8: "Cactus", 9: "Water"
        }


        from collections import OrderedDict
        self.pair_usage_order = OrderedDict()
        
        self.color_cache = {}
        self.pair_cache  = {}
        self.terminal_display_cache = {}

        self.vbo_initialized   = False
        self.last_vertex_count = 0
        
        self.update_camera_vectors()
        self._init_color_map()

        self.chat = ["Press '\\' to open the chat"]
        self.highlight_range   = 8.0  # reach distance
        self.highlighted_block = None
        self.highlighted_face  = None
        


        # mesh optimization
        # For low end hardware, please use buffer_update_interval=0.1 or 0.05
        # 0.1 value may cause flickering but its drasticly smoother
        self.vertex_buffer = None
        self.color_buffer  = None
        self.buffer_size   = 0
        self.last_buffer_update     = 0
        self.buffer_update_interval = 0.0  # 0.1  # 0.05
        
        # rendering optimizations
        self.frustum_culling   = True
        self.occlusion_culling = True
        self.view_frustum      = None
        self.last_frustum_update     = 0
        self.frustum_update_interval = 0.05
        
        # monitor
        self.frame_times = collections.deque(maxlen=30)  # 60 frames
        self.last_frame_time = time.time()
        
        
        self.setup_screen()
        self._init_buffers()



    def _init_color_map(self):
        self.xterm_colors = []
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    r_val = 0 if r == 0 else (40 + r * 40) if r < 5 else 255
                    g_val = 0 if g == 0 else (40 + g * 40) if g < 5 else 255
                    b_val = 0 if b == 0 else (40 + b * 40) if b < 5 else 255
                    self.xterm_colors.append((r_val, g_val, b_val))
    


    def rgb_to_color_index(self, r, g, b):
        key = (r, g, b)
        if key in self.color_cache:
            return self.color_cache[key]
        
        r_scaled, g_scaled, b_scaled = int(r * 255), int(g * 255), int(b * 255)
        best_idx, best_distance = 16, float('inf')
        
        for i, (cr, cg, cb) in enumerate(self.xterm_colors):
            distance = (cr - r_scaled)**2 + (cg - g_scaled)**2 + (cb - b_scaled)**2
            
            if distance < best_distance:
                best_distance = distance
                best_idx = i + 16
                
        self.color_cache[key] = best_idx
        return best_idx
    
    def update_camera_vectors(self):
        yaw_rad, pitch_rad = math.radians(self.yaw), math.radians(self.pitch)
        
        front = np.array(
            [
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad)
            ], dtype=np.float32
        )
        
        self.camera_front = front / np.sqrt(np.sum(front*front))
        
        right = np.cross(self.camera_front, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right = right / np.sqrt(np.sum(right*right))
        
        self.camera_up = np.cross(right, self.camera_front)



    def setup_gl_optimizations(self):
        # enable vbo
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # set hint for fast  render
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
        glHint(GL_FOG_HINT, GL_FASTEST)
        
        # disable these - no need
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        
        # face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
    
    def setup_screen(self):
        """init screen and opengl context"""
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(1)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        print("\033[?1003h")
        
        self.term_height, self.term_width = self.stdscr.getmaxyx()
        
        if curses.COLORS >= 256:
              self._init_color_pairs()
        else: self._init_basic_color_pairs()
        
        self.gl_width, self.gl_height = self.term_width, self.term_height * 2
        
        
        pygame.display.set_mode((self.gl_width, self.gl_height), DOUBLEBUF | OPENGL | HIDDEN)
        pygame.display.set_caption("Simulation")
        glViewport(0, 0, self.gl_width, self.gl_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.gl_width / self.gl_height, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        



        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        
        self.pixel_array = np.zeros((self.gl_height, self.gl_width, 3), dtype=np.uint8)
        
        self.display_width = min(self.term_width, self.gl_width)
        self.display_height = min(self.term_height, self.gl_height)
        #self.setup_gl_optimizations()
        


    def setup_vbos(self):
        if not self.vbo_initialized:
            self.vertex_vbo = glGenBuffers(1)
            self.color_vbo  = glGenBuffers(1)
            self.vbo_initialized = True
        
        # updade if vertices
        if hasattr(self.chunk_manager, 'vertices') and len(self.chunk_manager.vertices) > 0:
            # vertex
            glBindBuffer(GL_ARRAY_BUFFER,  self.vertex_vbo)
            glBufferData(GL_ARRAY_BUFFER,  self.chunk_manager.vertices.nbytes, self.chunk_manager.vertices, GL_DYNAMIC_DRAW)
            
            # color
            glBindBuffer(GL_ARRAY_BUFFER,  self.color_vbo)
            glBufferData(GL_ARRAY_BUFFER,  self.chunk_manager.colors.nbytes,   self.chunk_manager.colors, GL_DYNAMIC_DRAW)
            
            # store vertex count for drawing
            self.last_vertex_count   = len(self.chunk_manager.vertices) // 3
        else: self.last_vertex_count = 0
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    





    def _init_color_pairs(self):
        curses.init_pair(1, 94,  0)
        curses.init_pair(2, 28,  0)
        curses.init_pair(3, 244, 0)
        curses.init_pair(4, 94, 28)
        
        self.next_pair_id, self.max_pairs = 10, min(curses.COLOR_PAIRS - 1, 200) # 100
    
    def _init_basic_color_pairs(self):
        for bg in range(8):
            for fg in range(8):
                pair_idx = bg * 8 + fg + 1
                if pair_idx < 64:
                    curses.init_pair(pair_idx, fg, bg)
    


    def get_color_pair(self, top_color, bot_color):
        key = (top_color, bot_color)
        if key in self.pair_cache:
            # move to end (most recent)
            self.pair_usage_order.move_to_end(key)
            return self.pair_cache[key]
        
        if self.next_pair_id < self.max_pairs:
            try:
                curses.init_pair(self.next_pair_id, top_color, bot_color)
                self.pair_cache[key]       = self.next_pair_id
                self.pair_usage_order[key] = self.next_pair_id
                pair_id = self.next_pair_id
                self.next_pair_id += 1
                return pair_id
            except curses.error:
                pass
        
        # out of pairs -> reuse least recent
        if len(self.pair_usage_order) > 0:
            lru_key = next(iter(self.pair_usage_order))
            lru_pair_id = self.pair_usage_order[lru_key]
            
            # del old mapping
            del self.pair_cache[lru_key]
            del self.pair_usage_order[lru_key]
            
            # reuse the pair id for new colors
            try:
                curses.init_pair(lru_pair_id, top_color, bot_color)
                self.pair_cache[key]       = lru_pair_id
                self.pair_usage_order[key] = lru_pair_id
                return lru_pair_id
            except curses.error:
                pass
        
        return 1


    def set_selected_block(self, id):
        if self.selected_block != id:
            self.selected_block = id
            self.message(f"Selected item={self.block_names[id]}")
    
    def render_to_buffer(self):
        glReadBuffer(GL_BACK)
        pixel_data = glReadPixels(0, 0, self.gl_width, self.gl_height, GL_RGB, GL_UNSIGNED_BYTE)
        np.copyto(self.pixel_array, np.frombuffer(pixel_data, dtype=np.uint8).reshape(self.gl_height, self.gl_width, 3))
        self.pixel_array = np.flipud(self.pixel_array)
    



    def display_buffer(self):
        """Displays converted opengl render data into w*2h pixel art array using colored "▄" bg, fg """
        # TODO: save memory by using full block char aswell
        status_lines = 0  # nomore
        screen_changed = False
        
        # process data in batches
        for y in range(status_lines, self.display_height):
            y2 = (y - status_lines) * 2
            if y2 + 1 >= self.pixel_array.shape[0]:
                break
            
            
            top_row = self.pixel_array[y2+1, :self.display_width] / 255.0
            bot_row = self.pixel_array[y2,   :self.display_width] / 255.0
            
            # color -> index
            top_colors = [self.rgb_to_color_index(r, g, b) for r, g, b in top_row]
            bot_colors = [self.rgb_to_color_index(r, g, b) for r, g, b in bot_row]
            
            pairs = [self.get_color_pair(t, b) for t, b in zip(top_colors, bot_colors)]
            
            # check for dirty rows
            row_key = y
            pairs_tuple = tuple(pairs)
            if row_key not in self.terminal_display_cache or self.terminal_display_cache[row_key] != pairs_tuple:
                self.terminal_display_cache[row_key] = pairs_tuple
                screen_changed = True
                
                
                # drawing using run length encoding
                changes = np.diff(pairs, prepend=pairs[0]-1)
                change_indices = np.nonzero(changes)[0]
                
                for i in range(len(change_indices)):
                    start = change_indices[i]
                    end = change_indices[i+1] if i+1 < len(change_indices) else len(pairs)
                    try:  
                        # ▄ -> bg=top pixel, fg=bottom pixel
                        self.stdscr.addstr(y, start, '▄' * (end - start), curses.color_pair(pairs[start]))
                    except curses.error: pass
        
        self._draw_ui(screen_changed)

    
    def _draw_ui(self, screen_changed):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time   = current_time
            screen_changed   = True
        
        if screen_changed:
            # chunk
            pos = self.camera_position
            chunk_x, chunk_z = self.chunk_manager.get_chunk_coords_for_position(pos)
            stats = self.chunk_manager.stats
            
            # text overlay
            self.stdscr.addnstr(0, 0, f"FPS: {self.fps:.1f} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Yaw: {self.yaw:.1f}° Pitch: {self.pitch:.1f}°", self.term_width - 1)
            self.stdscr.addnstr(1, 0, f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Chunk: [{chunk_x}, {chunk_z}]",                                       self.term_width - 1)
            self.stdscr.addnstr(2, 0, f"Chunks: {stats['active_chunks']}/{stats['total_generated']} | Render Distance: {self.chunk_manager.render_distance}",   self.term_width - 1)
        
        for i in range(len(self.chat)):
            self.stdscr.addnstr(self.term_height - (len(self.chat)) + i, 0, self.chat[i], self.term_width - 1)

        # self.stdscr.addnstr(self.term_height // 2 - 1, self.term_width // 2 - 1, "x", 1)
        # text overlay should go here
        
        

    def draw_gl_ui(self):
        # save matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.gl_width, 0, self.gl_height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        # crosshair
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(1.0)
        
        center_x = self.gl_width / 2
        center_y = self.gl_height / 2
        
        glBegin(GL_LINES)
        glVertex2f(center_x - 2, center_y)
        glVertex2f(center_x + 1, center_y)

        glVertex2f(center_x, center_y - 1)
        glVertex2f(center_x, center_y + 2)
        glEnd()
        
        # hotbar
        total_width = (self.hotbar_slot_size + self.hotbar_padding) * self.hotbar_width - self.hotbar_padding
        start_x = (self.gl_width - total_width) / 2
        start_y = self.hotbar_y_offset

                
        # hotbar background
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(start_x - 1, start_y - 1)
        glVertex2f(start_x + total_width + 1, start_y - 1)
        glVertex2f(start_x + total_width + 1, start_y + self.hotbar_height + 1)
        glVertex2f(start_x - 1, start_y + self.hotbar_height + 1)
        glEnd()

        
        # slots
        for i in range(self.hotbar_width):
            slot_x = start_x + i * (self.hotbar_slot_size + self.hotbar_padding)
            
            # fuckass 3d effect
            if i == self.selected_block - 1:
                base_color   = (1.0, 1.0, 1.0, 0.5)
                bright_color = (1.0, 1.0, 1.0, 0.7)
            else:
                base_color   = (0.2, 0.2, 0.2, 0.3)
                bright_color = (0.5, 0.5, 0.5, 0.7)
            
            # main slot background
            glColor4f(*base_color)
            glBegin(GL_QUADS)
            glVertex2f(slot_x,  start_y)
            glVertex2f(slot_x + self.hotbar_slot_size, start_y)
            glVertex2f(slot_x + self.hotbar_slot_size, start_y + self.hotbar_height)
            glVertex2f(slot_x,  start_y + self.hotbar_height)
            glEnd()
            
            # bright edges
            glColor4f(*bright_color)
            glBegin(GL_LINES)
            # top
            glVertex2f(slot_x, start_y + self.hotbar_height)
            glVertex2f(slot_x + self.hotbar_slot_size, start_y + self.hotbar_height -1)
            # right
            glVertex2f(slot_x + self.hotbar_slot_size, start_y)
            glVertex2f(slot_x + self.hotbar_slot_size, start_y + self.hotbar_height)
            glEnd()
            
            # block color
            block_type = self.hotbar_blocks[i]
            if block_type != 0:
                color = self.block_colors[block_type]
                glColor3f(color[0], color[1], color[2])
                
                
                block_size = self.hotbar_slot_size * 0.8 # padding
                block_x = slot_x + (self.hotbar_slot_size - block_size) / 2
                block_y = start_y + (self.hotbar_height - block_size) / 2
                
                glBegin(GL_QUADS)
                glVertex2f(block_x, block_y)
                glVertex2f(block_x + block_size, block_y)
                glVertex2f(block_x + block_size, block_y + block_size)
                glVertex2f(block_x, block_y + block_size)
                glEnd()
        
        glEnable(GL_DEPTH_TEST)
        
        # restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glColor3f(1.0, 1.0, 1.0)

    


    def message(self, content):
        # limited chat thing
        # TODO actuall proper chat
        self.chat.append(content)
        if len(self.chat) > 5:
            self.chat.pop(0)

    def break_block(self):
        """break highlighted block"""
        if self.highlighted_block is None: return False
        
        block_x, block_y, block_z = self.highlighted_block
        
        # chunk coordinates
        chunk_x = block_x // self.chunk_manager.chunk_size
        chunk_z = block_z // self.chunk_manager.chunk_size
        chunk_coord = (chunk_x, chunk_z)
        
        if chunk_coord not in self.chunk_manager.chunks:
            return False
        
        # local coordinates within chunk
        local_x = block_x - chunk_x * self.chunk_manager.chunk_size
        local_z = block_z - chunk_z * self.chunk_manager.chunk_size
        
        if (local_x < 0 or local_x >= self.chunk_manager.chunk_size or
            block_y < 0 or block_y >= 64 or
            local_z < 0 or local_z >= self.chunk_manager.chunk_size):
            return False #invalid
        
        # tagret->air
        chunk = self.chunk_manager.chunks[chunk_coord]
        if chunk.blocks[local_x, block_y, local_z] != 0:
            chunk.blocks[local_x, block_y, local_z] = 0
            self.chunk_manager.update_chunk_mesh_fast(chunk_coord)
            self.message(f"Broken block at ({local_x},{block_y},{local_z})")
            return True
        
        return False



    def place_block(self):
        """place new block adjacent to highlighted face"""
        if self.highlighted_block is None or self.highlighted_face is None:
            return False
        
        block_x, block_y, block_z = self.highlighted_block
        
        # find placement position based on highlighted face
        # 0=front, 1=back, 2=top, 3=bottom, 4=Right, 5=Left
        face_offsets = [
            ( 0,  0,  1),  # front  +z
            ( 0,  0, -1),  # back   -z
            ( 0,  1,  0),  # top    +y
            ( 0, -1,  0),  # bottom -y
            ( 1,  0,  0),  # right  +x
            (-1,  0,  0)   # left   -x
        ]
        
        offset_x, offset_y, offset_z = face_offsets[self.highlighted_face]
        place_x = block_x + offset_x
        place_y = block_y + offset_y
        place_z = block_z + offset_z
        
        if place_y < 0 or place_y >= 64:  return False # invalid
        
        # find chunk coords
        chunk_x = place_x // self.chunk_manager.chunk_size
        chunk_z = place_z // self.chunk_manager.chunk_size
        chunk_coord = (chunk_x, chunk_z)
        
        if chunk_coord not in self.chunk_manager.chunks: return False # invalid: not loaded
        
        # find local coords
        local_x = place_x - chunk_x * self.chunk_manager.chunk_size
        local_z = place_z - chunk_z * self.chunk_manager.chunk_size
        


        if (local_x < 0 or local_x >= self.chunk_manager.chunk_size or
            local_z < 0 or local_z >= self.chunk_manager.chunk_size):
            return False # invalid
        
        # replace air with selected id
        chunk = self.chunk_manager.chunks[chunk_coord]
        if chunk.blocks[local_x, place_y, local_z] == 0:
            block_type = self.hotbar_blocks[self.selected_block - 1]
            chunk.blocks[local_x, place_y, local_z] = block_type
            self.chunk_manager.update_chunk_mesh_fast(chunk_coord)
            
            self.message(f"Placed block={self.block_names[block_type]} at ({local_x},{place_y},{local_z})")
            return True
        
        return False

    def raycast_to_block(self):
        """raycast from camera to find target block face"""
        ray_origin = self.camera_position.copy()
        ray_direction = self.camera_front.copy()
        """
        # TODO temporary fix to the offset, barely fine tuned, 
        # should be removed and fixed propperly, but works 4 now
        ray_origin[0] += abs(ray_direction[0] - 0.2 )
        ray_origin[1] += abs(ray_direction[1] - 0.2 )
        ray_origin[2] += abs(ray_direction[2] - 0.2 )
        #self.message(f"Ray dir: {ray_direction}")
        """
        ray_origin[0] += 0.5
        ray_origin[1] += 0.5
        ray_origin[2] += 0.5
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        step_size = 0.05  # 0.01
        max_steps = int(self.highlight_range / step_size)
        


        previous_block = None
        for i in range(max_steps):
            ray_pos = ray_origin + ray_direction * (i * step_size)
            
            # convert to coords
            block_x = int(np.floor(ray_pos[0]))
            block_y = int(np.floor(ray_pos[1]))
            block_z = int(np.floor(ray_pos[2]))
            
            current_block = (block_x, block_y, block_z)
            
            # find chunk
            chunk_x = block_x // self.chunk_manager.chunk_size
            chunk_z = block_z // self.chunk_manager.chunk_size
            
            if (chunk_x, chunk_z) not in self.chunk_manager.chunk_data:
                previous_block = current_block
                continue # not loaded, invalid
                
            # local coords
            local_x = block_x % self.chunk_manager.chunk_size
            local_z = block_z % self.chunk_manager.chunk_size
            

            # in bounds?
            if (local_x < 0 or local_x >= self.chunk_manager.chunk_size or
                block_y < 0 or block_y >= 64 or
                local_z < 0 or local_z >= self.chunk_manager.chunk_size):
                previous_block = current_block
                continue # out
                

                
            chunk_blocks = self.chunk_manager.chunk_data[(chunk_x, chunk_z)][0]
            block_type = chunk_blocks[local_x, block_y, local_z]
            

            if block_type != 0:
                # find face based on hitpoint
                if previous_block is not None:
                    prev_x, prev_y, prev_z = previous_block
                    dx = block_x - prev_x
                    dy = block_y - prev_y
                    dz = block_z - prev_z
                    
                    #                             # Entered from | Hit    | Result
                    if dx > 0:   face_idx = 5     # left         : right  : left
                    elif dx < 0: face_idx = 4     # right        : left   : right
                    elif dy > 0: face_idx = 3     # below        : top    : bottom 
                    elif dy < 0: face_idx = 2     # above        : bottom : top 
                    elif dz > 0: face_idx = 1     # behind       : front  : back 
                    elif dz < 0: face_idx = 0     # front        : back   : front 
                    else:        face_idx = 0     
                        
                        
                    return   (block_x, block_y, block_z), face_idx
                else: return (block_x, block_y, block_z), 0
            
            previous_block = current_block
        
        return None, None

    def draw_block_highlight(self):
        """wireframe highlight"""
        if self.highlighted_block is None or self.highlighted_face is None:
            return
            
        block_x, block_y, block_z = self.highlighted_block
        face_idx = self.highlighted_face
        
        # HERE
        # tuned for highlight
        face_vertices = [
            [[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]], # +z front
            [[-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5]], # -z back
            [[-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5]], # +y top
            [[-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5]], # -y bottom
            [[ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5]], # +x right
            [[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]], # -x left
        ]
        face_verts = np.array(face_vertices[face_idx]) + np.array([block_x, block_y, block_z])
        
        # expand slightly to avoid z fighting
        block_center = np.array([block_x, block_y, block_z])
        for i in range(4):
            offset = face_verts[i] - block_center
            face_verts[i] = block_center + offset * 1.002
        
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0) 

        glBegin(GL_LINE_LOOP)
        for vertex in face_verts:
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        glColor3f(1.0, 1.0, 1.0)



    def _init_buffers(self):
        self.vertex_buffer = glGenBuffers(1)
        self.color_buffer  = glGenBuffers(1)
        
        # prealloc buffer size
        initial_size = 1500000  # 800000  500000
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, initial_size * 3 * 4, None, GL_DYNAMIC_DRAW)  # 3 floats p vertex
        
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, initial_size * 3 * 4, None, GL_DYNAMIC_DRAW)  # 3 floats p color
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.buffer_size = initial_size

    def _update_buffers(self, vertices, colors):
        current_time = time.time()
        if current_time - self.last_buffer_update < self.buffer_update_interval: return
        self.last_buffer_update = current_time
        
        
        required_size = len(vertices) // 3
        
        
        if required_size > self.buffer_size:  # resize
            new_size = max(required_size * 2, self.buffer_size * 2)
            glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, new_size * 3 * 4, None, GL_DYNAMIC_DRAW)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
            glBufferData(GL_ARRAY_BUFFER, new_size * 3 * 4, None, GL_DYNAMIC_DRAW)
            
            self.buffer_size = new_size
        
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)



    def _update_view_frustum(self):
        current_time = time.time()
        if current_time - self.last_frustum_update < self.frustum_update_interval:
            return
        self.last_frustum_update = current_time

        modelview  = glGetFloatv(GL_MODELVIEW_MATRIX)
        projection = glGetFloatv(GL_PROJECTION_MATRIX)


        self.view_frustum = self._calculate_frustum_planes(modelview, projection)




    def _calculate_frustum_planes(self, modelview, projection):
        """find view frustum planes for culling"""
        mvp = np.dot(projection, modelview)
        planes = np.zeros((6, 4))
        
        planes[0] = mvp[3] + mvp[0]  # left
        planes[1] = mvp[3] - mvp[0]  # right
        planes[2] = mvp[3] + mvp[1]  # bottom
        planes[3] = mvp[3] - mvp[1]  # top

        planes[4] = mvp[3] + mvp[2]  # near
        planes[5] = mvp[3] - mvp[2]  # far
        
        
        for i in range(6):  # normalize
            length = np.sqrt(np.sum(planes[i, :3] * planes[i, :3]))
            planes[i] /= length  
            
        return planes

    def _is_in_frustum(self, position, radius):
        """test for frustum view bounds"""
        if not self.frustum_culling or self.view_frustum is None:
            return True
            
        for plane in self.view_frustum:
            distance = np.dot(plane[:3], position) + plane[3]
            if distance < -radius:
                return False
        return True

    def draw_scene(self):
        self._update_view_frustum()
        
        # sky bg
        glClearColor(0.529, 0.808, 0.922, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # update cameras
        look_at = self.camera_position + self.camera_front
        gluLookAt(
            self.camera_position[0], 
            self.camera_position[1], 
            self.camera_position[2],
            look_at[0], 
            look_at[1], 
            look_at[2],
            self.camera_up[0], 
            self.camera_up[1], 
            self.camera_up[2]
        )
        
        self.highlighted_block, self.highlighted_face = self.raycast_to_block()
        
        if hasattr(self.chunk_manager, 'vertices'): # mesh updates
            self._update_buffers(self.chunk_manager.vertices, self.chunk_manager.colors)
        
        # draw meshes
        if hasattr(self.chunk_manager, 'vertices') and len(self.chunk_manager.vertices) > 0:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
            glColorPointer(3, GL_FLOAT, 0, None)
            
            glDrawArrays(GL_QUADS, 0, len(self.chunk_manager.vertices) // 3)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        
        
        self.draw_block_highlight()
        


        # monitor
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        if len(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        pygame.display.flip()

    def cleanup(self):
        if hasattr(self, 'vertex_buffer'):  glDeleteBuffers(1, [self.vertex_buffer])
        if hasattr(self, 'color_buffer'):   glDeleteBuffers(1, [self.color_buffer])
        
        curses.endwin()
        print("\033[?1003l")
        pygame.quit()







