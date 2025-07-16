import time
import curses
import numpy as np
import threading
import collections
from typing import Set, Tuple, Optional
import logging

import keyboard
import pynput.mouse
import win32gui
import win32api

from render import TerminalRenderer
from chunk  import ThreadedChunkManager
from player import PlayerController
import title as titler


class InputManager:
    MAPPINGS = {
        'w': ord('w'), 's': ord('s'), 'a': ord('a'), 'd': ord('d'),
        'space': ord(' '), 'c': ord('c'), 'tab': 9, 'q': ord('q'),

        '1': ord('1'), '2': ord('2'), '3': ord('3'), '4': ord('4'),
        '5': ord('5'), '6': ord('6'), '7': ord('7'), '8': ord('8'),
        '9': ord('9'),

        'j': ord('j')
    }
    
    def __init__(self):
        self.keys_pressed:  Set[int] = set()
        self.mouse_buttons: Set[int] = set()
        
        self.mouse_clicks:  Set[int] = set()
        self.mouse_delta = np.zeros(2)
        self.lock = threading.Lock()
        self.running = True
        
        
        self.mouse_locked = False
        self.terminal_center = (0, 0)
        self.terminal_hwnd = None
        
        self._setup_listeners()
    
    def _setup_listeners(self):
        # keyboard listener
        for key, code in self.MAPPINGS.items():
            keyboard.on_press_key(key,   lambda e, k=code: self._on_key_press(k))
            keyboard.on_release_key(key, lambda e, k=code: self._on_key_release(k))
        
        # mouse listener
        self.mouse_listener = pynput.mouse.Listener(
            on_click=self._on_mouse_click,
            suppress=False
        )
        self.mouse_listener.start()
        
        # mouse lock thread
        self.lock_thread = threading.Thread(target=self._mouse_lock_loop, daemon=True)
        self.lock_thread.start()
    
    def _on_key_press(self, key_code: int):
        with self.lock: self.keys_pressed.add(key_code)
    
    def _on_key_release(self, key_code: int):
        with self.lock: self.keys_pressed.discard(key_code)
    

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool):
        button_map = {
            pynput.mouse.Button.left:   1,
            pynput.mouse.Button.right:  2,
            pynput.mouse.Button.middle: 3,
        }
        
        if button in button_map:
            button_id = button_map[button]
            with self.lock:
                if pressed:
                    self.mouse_buttons.add(button_id)
                    self.mouse_clicks.add(button_id)
                else: self.mouse_buttons.discard(button_id)
    
    def _find_window(self) -> bool:
        """find the terminal window and get its center"""
        hwnd = win32gui.GetForegroundWindow()
        rect = win32gui.GetWindowRect(hwnd)
        center_x = (rect[0] + rect[2]) // 2
        center_y = (rect[1] + rect[3]) // 2
        
        self.terminal_hwnd = hwnd
        self.terminal_center = (center_x, center_y)
        return True
    
    def _mouse_lock_loop(self, counter=0):
        """lock mouse to terminal center thread"""
        while self.running:
            if self.mouse_locked and self.terminal_hwnd:
                try:
                    # check window focus periodically
                    if counter % 30 == 0:
                        current_hwnd = win32gui.GetForegroundWindow()
                        if current_hwnd !=  self.terminal_hwnd:
                            continue
                    
                    # get mouse delta nd reset
                    current_pos = win32api.GetCursorPos()
                    dx = current_pos[0] - self.terminal_center[0]
                    dy = current_pos[1] - self.terminal_center[1]
                    
                    if dx != 0 or dy != 0:
                        with self.lock:
                            self.mouse_delta[0] += dx * 0.4
                            self.mouse_delta[1] += dy * 0.4
                        
                        win32api.SetCursorPos(self.terminal_center)
                        
                except Exception: self.mouse_locked = False
            
            counter += 1
            time.sleep(0.008)
    
    def lock_mouse(self):
        if self._find_window():
            self.mouse_locked = True
            win32api.SetCursorPos(self.terminal_center)
    
    def unlock_mouse(self):
        self.mouse_locked = False

    def toggle_lock(self):
        if self.mouse_locked:
            self.unlock_mouse()
        else: self.lock_mouse()

    # helds
    def get_keys(self) -> Set[int]:
        with self.lock:
            return self.keys_pressed.copy()
    def get_mouse_buttons(self) -> Set[int]:
        with self.lock:
            return self.mouse_buttons.copy()
    
    def get_mouse_clicks(self) -> Set[int]:
        """get single mouse clicks (1time events)"""
        with self.lock:
            clicks = self.mouse_clicks.copy()
            self.mouse_clicks.clear()
            return clicks
    
    def get_mouse_delta(self) -> Tuple[float, float]:
        """get and reset mouse movement delta"""
        with self.lock:
            delta = tuple(self.mouse_delta * 0.85)
            self.mouse_delta *= 0.1
            return delta
    
    def cleanup(self):
        self.running = False
        self.unlock_mouse()
        keyboard.unhook_all()
        self.mouse_listener.stop()
        if self.lock_thread.is_alive():
            self.lock_thread.join(timeout=1.0)


class ChunkUpdateManager:

    
    def __init__(self, chunk_manager, 
    update_interval=1.0, min_movement_threshold=6.0, min_fps_threshold=15.0):
        self.chunk_manager    = chunk_manager
        self.last_update_time = 0
        self.last_chunk_pos   = None
        self.last_camera_pos  = np.zeros(3)
        self.position_history = collections.deque(maxlen=3)   # maxlen=5
        
        # configs moved to parent controller class            # (old)
        self.update_interval        = update_interval         # 0.7
        self.min_movement_threshold = min_movement_threshold  # 4.0 
        self.min_fps_threshold      = min_fps_threshold       # 20.0

        
    
    def should_update(self, camera_pos: np.ndarray, current_fps: float) -> bool:

        current_time = time.time()
        current_chunk_pos = self.chunk_manager.get_chunk_coords_for_position(camera_pos)
        
        # critical update for new chunk ( player pos in chunk )
        if self.last_chunk_pos != current_chunk_pos:
            self._update_tracking(current_time, current_chunk_pos, camera_pos)
            return True
        
        
        if current_fps < self.min_fps_threshold: return False # min_fps_threshold
        
        # check if player has moved enough to warrant an update
        time_elapsed   = current_time - self.last_update_time > self.update_interval
        distance_moved = np.linalg.norm(camera_pos - self.last_camera_pos) > self.min_movement_threshold
        
        if time_elapsed or distance_moved:
            self._update_tracking(current_time, current_chunk_pos, camera_pos)
            return True
        

        return False
    
    def _update_tracking(self, current_time: float, chunk_pos, camera_pos: np.ndarray):
        self.last_update_time = current_time
        self.last_chunk_pos   = chunk_pos
        self.last_camera_pos  = np.copy(camera_pos)
        self.position_history.append(np.copy(camera_pos))
    
    def update_chunks(self, camera_pos: np.ndarray):
        # share position history for predictive loading
        if hasattr(self.chunk_manager, 'position_history'):
            self.chunk_manager.position_history = list(self.position_history)[-3:]
        
        self.chunk_manager.update_chunks_around_position(camera_pos)


class GameController:

    
    def __init__(self, update_interval=1.0, render_distance=3, min_fps_threshold=5.0, target_fps=60.0, world_name=None, texture_name="basic"):
        """
        For non-dev users, 
        If you have a medium-high end hardware, adjust the following settings:
        Configuration settings:
          NAME                    LOW END     MEDIUM END     HIGH END   DESCRIPTION
        * update_interval:        1.0         0.5            0.2        time buffer between chunk updates
        * min_movement_threshold: 8.0         5.0            3.0        minimum walked distance to trigger an update
        * min_fps_threshold:      25.0        20.0           10.0       minimum frames per second to trigger an update

        * render_distance:        3           5              7          number of chunks to render around the player
        * max_workers:            6           8              12         maximum number of worker threads for generation
        """
        logging.info("The following settings are designed for low end hardware. Adjust as needed.")
        # basic chunk update/gen time downgrade for smooth controls
        # adjust based on hardware  # slow gen     # fast gen
        self.update_interval        = update_interval          # 0.2
        self.render_distance        = render_distance          # 3
        self.min_fps_threshold      = min_fps_threshold        # 15.0
        
        self._log_specs(self.__dict__.items())
        
        self.chunk_manager = ThreadedChunkManager(
            chunk_size      = 16, # 16  # chunk size
            render_distance = self.render_distance,  # x chunk radius loaded to render
            max_workers     = 6,  # 10  # x worker threads for chunk generation
            world_name      = world_name,
            texture_name    = texture_name
        )
        self.renderer = TerminalRenderer(self.chunk_manager)
        self.chunk_manager.set_log_callback(self.renderer.message) # TODO

        self.player = PlayerController(self.chunk_manager)
        self.input_manager = InputManager()
        self.chunk_updater = ChunkUpdateManager(
            self.chunk_manager,
            self.update_interval,
            6.0,  # min_movement_threshold - keeping default value
            self.min_fps_threshold
        )
        
        
        self.running = True
        self.last_update_time = time.time()
        self._tab_was_pressed = False
        self._last_mouse_lock_toggle = 0  # Add this line for debounce

        self.target_fps = target_fps
        self.smooth_fps = target_fps

        self.pause_menu_active = False
        self.pause_menu_buttons = ["Resume", "Quit"]
        self.pause_menu_selected = 0  # 0: Resume, 1: Quit

    def _log_specs(self, items, out=[]):
        for k, v in items: #self.__dict__.items():
            if k.startswith("_"): continue
            out.append(f"{k.title()}={v}")
        logging.info("Chunk Updates: " + " ".join(out))
    
    def _get_mouse_gl_coords(self):
        mouse_x, mouse_y = win32api.GetCursorPos()
        if self.input_manager.terminal_hwnd:
            # client area -> coordinates
            client_pt = win32gui.ScreenToClient(self.input_manager.terminal_hwnd, (mouse_x, mouse_y))
            client_x, client_y = client_pt
            # size
            left, top, right, bottom = win32gui.GetClientRect(self.input_manager.terminal_hwnd)
            client_w = right - left
            client_h = bottom - top
            # map to opengl size
            gl_mouse_x = int(client_x * self.renderer.gl_width / client_w)
            gl_mouse_y = int((client_h - client_y) * self.renderer.gl_height / client_h)
            return gl_mouse_x, gl_mouse_y+2
        return -1, -1

    def _handle_input(self):
        
        if self.pause_menu_active:
            keys = self.input_manager.get_keys()
            mouse_clicks = self.input_manager.get_mouse_clicks()
            gl_mouse_x, gl_mouse_y = self._get_mouse_gl_coords()

            # mouse hover
            hover_idx = self.renderer.get_pause_menu_hovered(gl_mouse_x, gl_mouse_y, len(self.pause_menu_buttons))
            if hover_idx is not None: self.pause_menu_selected = hover_idx


            # mous click
            if 1 in mouse_clicks and hover_idx is not None:
                if hover_idx == 1:
                    self.pause_menu_active = False
                    self.input_manager.lock_mouse()
                    self.renderer.show_block_name = True
                elif hover_idx == 0:
                    self.running = False
                return
            
            # pause screen
            if self.input_manager.MAPPINGS['q'] in keys:
                
                now = time.time()
                if not hasattr(self, '_last_mouse_lock_toggle'):
                    self._last_mouse_lock_toggle = 0
                if now - self._last_mouse_lock_toggle > 0.5:
                    self.pause_menu_active = False
                    self.input_manager.lock_mouse()
                    self._last_mouse_lock_toggle  = now
                    self.renderer.show_block_name = True
                return
            return


        # -- normal input --
        keys = self.input_manager.get_keys()
        mouse_buttons = self.input_manager.get_mouse_buttons()
        mouse_clicks  = self.input_manager.get_mouse_clicks()
        # pause toggle
        if self.input_manager.MAPPINGS['q'] in keys:
            now = time.time()
            if not hasattr(self, '_last_mouse_lock_toggle'):
                self._last_mouse_lock_toggle  = 0
                self.renderer.show_block_name = True
            if now - self._last_mouse_lock_toggle > 0.5:
                self.pause_menu_active = True
                self.input_manager.unlock_mouse()
                self._last_mouse_lock_toggle  = now
                self.renderer.show_block_name = False
            return
        
        """
        if self.input_manager.MAPPINGS['q'] in keys:
            self.running = False
            return
        """
        
        """
        # Clear modifications (for testing)
        if self.input_manager.MAPPINGS['q'] in keys:
            self.chunk_manager.clear_all_modifications()
            self.renderer.message("Cleared all modifications")
            return
        """
        
        # hotbar
        for i in range(1, 10):
            if self.input_manager.MAPPINGS[str(i)] in keys:
                #self.renderer.selected_block = i
                self.renderer.set_selected_block(i)
                break
        
        # mouse look
        dx, dy = self.input_manager.get_mouse_delta()
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self.renderer.yaw += dx * self.renderer.mouse_sensitivity
            self.renderer.pitch -= dy * self.renderer.mouse_sensitivity
            self.renderer.pitch = np.clip(self.renderer.pitch, -89.0, 89.0)
            self.renderer.update_camera_vectors()
        


        if 1 in mouse_clicks: self.renderer.break_block()
        if 2 in mouse_clicks: self.renderer.place_block()
        
        # keyboard move
        self.player.moving_forward  = self.input_manager.MAPPINGS['w'] in keys
        self.player.moving_backward = self.input_manager.MAPPINGS['s'] in keys
        self.player.moving_left     = self.input_manager.MAPPINGS['a'] in keys
        self.player.moving_right    = self.input_manager.MAPPINGS['d'] in keys
        
        self.player.moving_down     = self.input_manager.MAPPINGS['c'] in keys
        self.player.moving_up       = self.input_manager.MAPPINGS['space'] in keys
        self.player.jumping         = self.input_manager.MAPPINGS['space'] in keys

        
        if self.input_manager.MAPPINGS['tab'] in keys:  # toggle flight
            if not self._tab_was_pressed:
                self.player.toggle_flight_mode()
                self._tab_was_pressed = True
        else:   self._tab_was_pressed = False
    
    def _update_player(self):
        """update position physics thing"""
        current_time = time.time()
        delta_time   = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # update player
        self.player.update(
            delta_time,
            self.renderer.camera_front,
            self.renderer.camera_up,
            self.input_manager.get_keys()
        )
        
        # cam follow
        self.renderer.camera_position = np.copy(self.player.position)
        self.renderer.camera_position[1] += 1.2  # eye height, should be < player height
    
    def _update_chunks(self):
        """update chunks if needed"""
        if  self.chunk_updater.should_update(self.renderer.camera_position, self.smooth_fps):
            self.chunk_updater.update_chunks(self.renderer.camera_position)
        
        # TODO: optimize chunk unloading, causing lag spikes
        if hasattr(self.chunk_manager, 'chunks_to_remove_from_mesh') and self.chunk_manager.chunks_to_remove_from_mesh:
            self.chunk_manager.update_mesh_after_unload()
            
    def _render_frame(self):


        if hasattr(self.chunk_manager, 'process_completed_meshes'):
            self.chunk_manager.process_completed_meshes()
        
        
        
        self.renderer.draw_scene()
        if self.pause_menu_active:
            gl_mouse_x, gl_mouse_y = self._get_mouse_gl_coords()
            hover_idx = self.renderer.get_pause_menu_hovered(gl_mouse_x, gl_mouse_y, len(self.pause_menu_buttons))
            self.renderer.draw_pause_menu(self.pause_menu_selected, self.pause_menu_buttons, mouse_hover_idx=hover_idx, mouse_pos=(gl_mouse_x, gl_mouse_y))
        elif self.renderer.toggle_ui:
            self.renderer.draw_gl_ui()
        self.renderer.render_to_buffer()
        self.renderer.display_buffer()
        curses.doupdate()

    
    def _calculate_fps(self, frame_time: float) -> float:
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.smooth_fps = self.smooth_fps * 0.95 + current_fps * 0.05
        return self.smooth_fps
    
    def run(self):
        self.renderer.setup_screen()
        
        threading.Timer(0.1, self.input_manager.lock_mouse).start()
        try:
            target_frame_time = 1.0 / self.target_fps
            self.renderer.camera_position = np.copy(self.player.position)
                
            while self.running:
                frame_start = time.time()
                

                self.renderer.stdscr.nodelay(True)
                self.renderer.stdscr.getch()
                
                self._handle_input()
                if not self.running: break
                    
                self._update_player()
                self._update_chunks()
                self._render_frame()
                
                
                frame_time = time.time() - frame_start
                self._calculate_fps(frame_time)
                
                
                sleep_time = max(0, target_frame_time - frame_time)
                if sleep_time > 0:  time.sleep(sleep_time)
                
        except KeyboardInterrupt: self.running = False
        finally:  self._cleanup()
    
    def _cleanup(self):
        if hasattr(self.chunk_manager, 'chunk_modifications'):
            for chunk_coord in self.chunk_manager.chunk_modifications:
                self.chunk_manager.save_chunk_modifications(chunk_coord)
            logging.info("Saved all chunk modifications")
        #self.chunk_manager.clear_all_modifications()
        #logging.info("Cleared all chunk modifications")
        self.input_manager.cleanup()
        self.chunk_manager.cleanup()
        self.renderer.cleanup()

# -- color display --
# if you're developing a texture, use this to get all available values!
def xterm_index_to_rgb(index):
    if index < 16:
        # standard colors, you can look up a table
        return (0.0, 0.0, 0.0)
    index -= 16
    r = (index // 36) % 6
    g = (index // 6) % 6
    b = index % 6

    def level(n):
        return 0 if n == 0 else 95 + 40 * (n - 1)

    rf = level(r) / 255.0
    gf = level(g) / 255.0
    bf = level(b) / 255.0


    # round 0.01
    import math
    rf = math.ceil(rf * 100) / 100.0
    gf = math.ceil(gf * 100) / 100.0
    bf = math.ceil(bf * 100) / 100.0
    return (rf, gf, bf)



def print_xterm_color_palette():
    # print all xterm colors
    for i in range(16):
        for j in range(16):
            color = i * 16 + j
            print(f"\033[48;5;{color}m ", end="")
        print("\033[0m", end="")
    print("\033[0m")
    # NOTE: Uncomment this to display all color values
    """for i in range(16):
        for j in range(16):
            color = i * 16 + j
            print(f"\033[48;5;{color}m RGB: {xterm_index_to_rgb(color)}", end="")
        print(f"\033[0m", end="")
    print("\033[0m")
"""


if __name__ == "__main__":
    print("    -- NEW INSTANCE -- ")
    import multiprocessing
    multiprocessing.freeze_support()
    import argparse
    import json
    from pathlib import Path

    # args for laucher
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_update_intreval", type=float, default= 1.0    )
    parser.add_argument("--render_distance",       type=int,   default= 3      )
    parser.add_argument("--min_fps_threshold",     type=float, default= 5.0    )
    parser.add_argument("--target_fps",            type=float, default=  60.0  )
    parser.add_argument("--texture",               type=str,   default= "basic")
    parser.add_argument("--world",                 type=str,   default= None   )
    args = parser.parse_args()
    

    print_xterm_color_palette()
    titler.generate()
    game = GameController(
        update_interval=args.chunk_update_intreval,
        render_distance=args.render_distance,
        min_fps_threshold=args.min_fps_threshold,
        target_fps=args.target_fps,
        world_name=args.world,
        texture_name=args.texture
    )
    game.run()
