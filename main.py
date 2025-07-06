import time
import curses
import sys
import numpy as np
import threading
import signal
import collections
from typing import Set, Tuple, Optional
import logging

import pynput.mouse
if sys.platform == "win32":
    import keyboard
    import win32gui
    import win32api
else:
    # Assume X11
    from Xlib import display, X
    import Xlib.XK
    d = display.Display()
    s = d.screen()
    root = s.root

from render import TerminalRenderer
from chunk  import ThreadedChunkManager
from player import PlayerController


class InputManager:
    MAPPINGS = {
        'w': ord('w'), 's': ord('s'), 'a': ord('a'), 'd': ord('d'),
        'space': ord(' '), 'c': ord('c'), 'tab': 9, 'q': ord('q'),
        '1': ord('1'), '2': ord('2'), '3': ord('3'), '4': ord('4'),
        '5': ord('5'), '6': ord('6'), '7': ord('7'), '8': ord('8'),
        '9': ord('9'),
        'up': -1, 'down': -2, 'left': -3, 'right': -4,
    }

    def __init__(self):
        self.keys_pressed: Set[int] = set()
        self.mouse_buttons: Set[int] = set()
        self.mouse_clicks: Set[int] = set()
        self.mouse_delta = np.zeros(2)
        self.lock = threading.Lock()
        self.running = True

        self.mouse_locked = False
        self.terminal_center = (200, 200)
        self.terminal_hwnd = None

        self._setup_listeners()

    def _setup_listeners(self):
        if sys.platform == "win32":
            for key, code in self.MAPPINGS.items():
                keyboard.on_press_key(key,   lambda e, k=code: self._on_key_press(k))
                keyboard.on_release_key(key, lambda e, k=code: self._on_key_release(k))
            keyboard.on_press_key("esc", lambda e: self._exit())
            keyboard.on_press_key("ctrl+c", lambda e: self._exit())
        else:
            self.keyboard_thread = threading.Thread(target=self._x11_input_loop, daemon=True)
            self.keyboard_thread.start()

        self.mouse_listener = pynput.mouse.Listener(
            on_click=self._on_mouse_click,
            suppress=False
        )
        self.mouse_listener.start()

        self.lock_thread = threading.Thread(target=self._mouse_lock_loop, daemon=True)
        self.lock_thread.start()

    def _on_key_press(self, key_code: int):
        with self.lock:
            self.keys_pressed.add(key_code)
        if key_code in (3, 27):  # CTRL+C or ESC
            self._exit()

    def _on_key_release(self, key_code: int):
        with self.lock:
            self.keys_pressed.discard(key_code)

    def _x11_input_loop(self):
        keysyms_map = {}
        for k, v in self.MAPPINGS.items():
            sym = Xlib.XK.string_to_keysym(k)
            if sym:
                keysyms_map[sym] = v
        keysyms_map[Xlib.XK.XK_Up] = -1
        keysyms_map[Xlib.XK.XK_Down] = -2
        keysyms_map[Xlib.XK.XK_Left] = -3
        keysyms_map[Xlib.XK.XK_Right] = -4

        # Also watch for raw ESC and Ctrl+C
        keysyms_map[Xlib.XK.XK_Escape] = 27
        keysyms_map[Xlib.XK.XK_C] = ord('c')

        root.grab_keyboard(True, X.GrabModeAsync, X.GrabModeAsync, X.CurrentTime)
        root.grab_pointer(True, X.PointerMotionMask | X.ButtonPressMask | X.ButtonReleaseMask,
                          X.GrabModeAsync, X.GrabModeAsync, X.NONE, X.NONE, X.CurrentTime)

        while self.running:
            while d.pending_events():
                e = d.next_event()
                if e.type == X.KeyPress or e.type == X.KeyRelease:
                    keysym = d.keycode_to_keysym(e.detail, 0)
                    key_code = keysyms_map.get(keysym)
                    if key_code is not None:
                        if key_code in (3, 27):
                            self._exit()
                        elif key_code in (-1, -2, -3, -4):
                            with self.lock:
                                if key_code == -1: self.mouse_delta[1] -= 3
                                elif key_code == -2: self.mouse_delta[1] += 3
                                elif key_code == -3: self.mouse_delta[0] -= 3
                                elif key_code == -4: self.mouse_delta[0] += 3
                        else:
                            if e.type == X.KeyPress:
                                self._on_key_press(key_code)
                            else:
                                self._on_key_release(key_code)
                elif e.type == X.MotionNotify:
                    dx = e.root_x - self.terminal_center[0]
                    dy = e.root_y - self.terminal_center[1]
                    if dx != 0 or dy != 0:
                        with self.lock:
                            self.mouse_delta[0] += dx * 0.4
                            self.mouse_delta[1] += dy * 0.4
                        root.warp_pointer(self.terminal_center[0], self.terminal_center[1])
                        d.sync()
                elif e.type in (X.ButtonPress, X.ButtonRelease):
                    btn = e.detail
                    with self.lock:
                        if e.type == X.ButtonPress:
                            self.mouse_buttons.add(btn)
                            self.mouse_clicks.add(btn)
                        else:
                            self.mouse_buttons.discard(btn)
            time.sleep(0.001)

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool):
        if sys.platform != "win32":
            return
        button_map = {
            pynput.mouse.Button.left: 1,
            pynput.mouse.Button.right: 2,
            pynput.mouse.Button.middle: 3,
        }
        if button in button_map:
            button_id = button_map[button]
            with self.lock:
                if pressed:
                    self.mouse_buttons.add(button_id)
                    self.mouse_clicks.add(button_id)
                else:
                    self.mouse_buttons.discard(button_id)

    def _find_window(self) -> bool:

        # Fallback cursor default location
        center_x = 200
        center_y = 200

        if sys.platform == "win32":
            hwnd = win32gui.GetForegroundWindow()
            rect = win32gui.GetWindowRect(hwnd)
            center_x = (rect[0] + rect[2]) // 2
            center_y = (rect[1] + rect[3]) // 2
        else:
            hwnd = d.get_input_focus().focus
            geom = hwnd.get_geometry()
            win_x, win_y = hwnd.translate_coords(root, 0, 0)
            rect = (win_x, win_y, win_x + geom.width, win_y + geom.height)

        self.terminal_hwnd = hwnd
        self.terminal_center = (center_x, center_y)
        return True

    def _mouse_lock_loop(self, counter=0):
        while self.running:
            if self.mouse_locked and self.terminal_hwnd:
                try:
                    if counter % 30 == 0:
                        if sys.platform == "win32":
                            current_hwnd = win32gui.GetForegroundWindow()
                        else:
                            current_hwnd = d.get_input_focus().focus
                        if current_hwnd != self.terminal_hwnd:
                            continue
                    if sys.platform == "win32":
                        current_pos = win32api.GetCursorPos()
                        dx = current_pos[0] - self.terminal_center[0]
                        dy = current_pos[1] - self.terminal_center[1]
                        if dx != 0 or dy != 0:
                            with self.lock:
                                self.mouse_delta[0] += dx * 0.4
                                self.mouse_delta[1] += dy * 0.4
                            win32api.SetCursorPos(self.terminal_center)
                except Exception:
                    self.mouse_locked = False
            counter += 1
            time.sleep(0.008)

    def lock_mouse(self):
        if self._find_window():
            self.mouse_locked = True
            if sys.platform == "win32":
                win32api.SetCursorPos(self.terminal_center)
            else:
                root.warp_pointer(200, 200)
                d.sync()

    def unlock_mouse(self):
        self.mouse_locked = False

    def get_keys(self) -> Set[int]:
        with self.lock:
            return self.keys_pressed.copy()

    def get_mouse_buttons(self) -> Set[int]:
        with self.lock:
            return self.mouse_buttons.copy()

    def get_mouse_clicks(self) -> Set[int]:
        with self.lock:
            clicks = self.mouse_clicks.copy()
            self.mouse_clicks.clear()
            return clicks

    def get_mouse_delta(self) -> Tuple[float, float]:
        with self.lock:
            delta = tuple(self.mouse_delta * 0.85)
            self.mouse_delta *= 0.1
            return delta

    def _exit(self):
        print("Exiting on input.")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.running = False
        self.unlock_mouse()
        if sys.platform == "win32":
            keyboard.unhook_all()
        else:
            root.ungrab_keyboard(X.CurrentTime)
            root.ungrab_pointer(X.CurrentTime)
            d.flush()
        self.mouse_listener.stop()
        if self.lock_thread.is_alive():
            self.lock_thread.join(timeout=1.0)
        if not sys.platform == "win32" and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1.0)


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

    
    def __init__(self):
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
        self.update_interval        = 1.0          # 0.2
        self.min_movement_threshold = 6.0          # 4.0
        self.min_fps_threshold      = 20.0         # 15.0
        
        self._log_specs(self.__dict__.items())
        
        self.chunk_manager = ThreadedChunkManager(
            chunk_size      = 16, # 16  # chunk size
            render_distance = 3,  # 5   # x chunk radius loaded to render
            max_workers     = 6   # 10  # x worker threads for chunk generation
        )
        self.renderer = TerminalRenderer(self.chunk_manager)
        self.chunk_manager.set_log_callback(self.renderer.message) # TODO

        self.player = PlayerController(self.chunk_manager)
        self.input_manager = InputManager()
        self.chunk_updater = ChunkUpdateManager(
            self.chunk_manager,
            self.update_interval,
            self.min_movement_threshold,
            self.min_fps_threshold
        )
        
        
        self.running = True
        self.last_update_time = time.time()
        self._tab_was_pressed = False

        self.target_fps = 60.0
        self.smooth_fps = 60.0

    def _log_specs(self, items, out=[]):
        for k, v in items: #self.__dict__.items():
            if k.startswith("_"): continue
            out.append(f"{k.title()}={v}")
        logging.info("Chunk Updates: " + " ".join(out))
    
    def _handle_input(self):
        
        keys = self.input_manager.get_keys()
        mouse_buttons = self.input_manager.get_mouse_buttons()
        mouse_clicks = self.input_manager.get_mouse_clicks()
        
        """
        if self.input_manager.MAPPINGS['q'] in keys:
            self.running = False
            return
        """
        
        # hotbar
        for i in range(1, 10):
            if self.input_manager.MAPPINGS[str(i)] in keys:
                #self.renderer.selected_block = i
                self.renderer.set_selected_block(i)
                """block_names = {
                    1: "Grass",  2: "Dirt",   3: "Stone",
                    4: "Log",    5: "Wood",   6: "Leaves",
                    7: "Sand",   8: "Cactus", 9: "Water",
                }
                self.renderer.message(f"Selected {block_names[i]}")"""
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
        self.input_manager.cleanup()
        self.chunk_manager.cleanup()
        self.renderer.cleanup()


def main():
    game = GameController()
    game.run()



if __name__ == "__main__":
    import title
    main()
