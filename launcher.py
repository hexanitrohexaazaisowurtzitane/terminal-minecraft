import curses
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pygame
import time
from PIL import Image
import subprocess
import sys
import os
import win32gui
import win32api
import random
import keyboard
import threading
import json

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ImageTerminalRenderer:
    def __init__(self, image_path="bg.png", title_path="title.png", title_top_padding=20):
        pygame.init()
        
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(1)
        
        self.term_height, self.term_width = self.stdscr.getmaxyx()
        self.gl_width,    self.gl_height  = self.term_width + 1, self.term_height * 2 + 1
        
        pygame.display.set_mode((self.gl_width, self.gl_height), DOUBLEBUF | OPENGL | HIDDEN)
        glViewport(0, 0, self.gl_width, self.gl_height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.gl_width, 0, self.gl_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        self.texture_id = self.load_texture(image_path, is_title=False)
        self.title_texture_id = self.load_texture(title_path, is_title=True)
        
        self._init_color_system()
        
        self.pixel_array = np.zeros((self.gl_height, self.gl_width, 3), dtype=np.uint8)
        
        self.title_width  = 0
        self.title_height = 0
        self.title_x = 0
        self.title_y = 0
        self.title_top_padding = title_top_padding
        self.running = True
        
        if self.title_texture_id: self._calculate_title_position()
        
        self.ui_buttons = ["Quit", "Settings", "Play"]
        self.ui_selected_button = 2
        self.ui_button_width    = 35
        self.ui_button_height   = 7
        self.ui_button_spacing  = 3
        self.ui_button_color          = (0.5, 0.5, 0.5)
        self.ui_button_selected_color = (0.7, 0.7, 0.7)
        
        
        self.mouse_x = 0
        self.mouse_y = 0
        self.terminal_hwnd    = None
        self.last_mouse_state = False
        
        self.active_menu = None
        self.menu_selected_button = 0



        # --- new world menu state ---
        
        self.nw_world_seed = str(random.randint(10000, 99999))
        self.nw_world_name = f"world_{self.nw_world_seed}"
        self.nw_generate_features = True
        self.nw_generate_caves    = True
        self.nw_selected_field    = 0  # 0: name, 1: seed, 2: features, 3: caves
        self.nw_editing           = False
        self._edit_buffer         = ""
        self._edit_field          = None  # 0 for name, 1 for seed
        self._edit_lock           = threading.Lock()
        self._keyboard_listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_listener_thread.start()


        # --- load world menu state ---
        self.lw_worlds      = self._scan_saves_directory()
        self.lw_num_buttons = max(len(self.lw_worlds), 1)
        self.lw_selected_button = 0
        self.lw_scroll_offset   = 0  # 4 worlds on display
        self.lw_visible_count   = 4  # 
        self.lw_scroll_up_hovered   = False
        self.lw_scroll_down_hovered = False
        self.lw_current_selected = 0 if len(self.lw_worlds) > 0 else -1
        
        # --- settings menu state ---
        self.settings_config = self._load_config()
        self.settings_sliders = [
            {"name": "Chunk Update Interval", "value": self.settings_config.get("chunk_update_intreval", 1.0), "min": 0.5,  "max": 5.0,   "step": 0.5},
            {"name": "Render Distance",       "value": self.settings_config.get("render_distance",         3), "min": 2,    "max": 10,    "step":   1},
            {"name": "Target FPS",            "value": self.settings_config.get("target_fps",           60.0), "min": 30.0, "max": 120.0, "step": 5.0}
        ]
        self.settings_selected_widget = 0  # 0-3: sliders, 4: textures button
        self.settings_dragging_slider = False
        self.settings_drag_start_x = 0
        self.settings_drag_start_value = 0.0
        
        # --- textures menu state ---
        self.textures_list        = self._scan_textures_directory()
        self.textures_num_buttons = max(len(self.textures_list), 1)
        self.textures_selected_button = 0
        self.textures_scroll_offset   = 0  # 4 display
        self.textures_visible_count   = 4  # 
        self.textures_scroll_up_hovered   = False
        self.textures_scroll_down_hovered = False
        
        texture_name_from_config = self.settings_config.get("texture", None)
        self.textures_current_selected = 0 if len(self.textures_list) > 0 else -1  # Default to first texture
        if texture_name_from_config is not None:
            for i, tex in enumerate(self.textures_list):
                if tex["name"] == texture_name_from_config:
                    self.textures_current_selected = i
                    break
    
    def _calculate_title_position(self):
        try:
            title_img    = Image.open(resource_path("title.png"))
            title_aspect = title_img.width / title_img.height
            
            title_width  = self.term_width - 6
            title_width  = int(title_width * 2 * 0.5)
            title_height = int(title_width / title_aspect)
            
            self.title_width  = title_width
            self.title_height = title_height
            self.title_x = (self.gl_width - title_width) // 2
            self.title_y = self.gl_height - title_height - self.title_top_padding
            
        except Exception as e:
            self.title_width = self.gl_width // 2
            self.title_height = self.gl_height // 4
            self.title_x = (self.gl_width - self.title_width) // 2
            self.title_y = self.gl_height - self.title_height - self.title_top_padding
    
    def _scan_saves_directory(self, saves_dir="saves", worlds=[]):
        if not os.path.exists(saves_dir): return worlds
        
        try:
            for item in os.listdir(saves_dir):
                world_path = os.path.join(saves_dir, item)
                if os.path.isdir(world_path):
                    # total folder size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(world_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:  total_size += os.path.getsize(filepath)
                            except (OSError, IOError): pass
                    
                    # format size for display
                    if total_size < 1024:           size_str = f"{total_size} B"
                    elif total_size < 1024 * 1024:  size_str = f"{total_size // 1024} KB"
                    else:  size_str = f"{total_size // (1024 * 1024)} MB"
                    
                    # world json settings -> seed
                    data_json_path = os.path.join(world_path, "data.json")
                    world_seed = "no data.json found"
                    
                    if os.path.exists(data_json_path):
                        try:
                            with open(data_json_path, 'r') as f:
                                data = json.load(f)
                                if 'seed' in data:
                                    world_seed   = str(data['seed'])
                                else: world_seed = "no seed in data.json"
                        except (json.JSONDecodeError, IOError, OSError):
                            world_seed = "invalid data.json"
                    
                    worlds.append({
                        'name': item,
                        'seed': world_seed,
                        'size': size_str,
                        'path': world_path
                    })
            
            
            worlds.sort(key=lambda x: x['name'].lower())
            
        except (OSError, IOError) as e:  pass
        return worlds


    
    def _scan_textures_directory(self, textures_dir="texture", textures=[]):
        if not os.path.exists(textures_dir): return textures
        
        try:
            for item in os.listdir(textures_dir):
                texture_path = os.path.join(textures_dir, item)
                if os.path.isfile(texture_path) and item.endswith('.json'):


                    try: file_size = os.path.getsize(texture_path)
                    except (OSError, IOError): file_size = 0
                    
                    # format size for display
                    if file_size < 1024:           size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:  size_str = f"{file_size // 1024} KB"
                    else:   size_str = f"{file_size // (1024 * 1024)} MB"
                    
                    

                    texture_name = os.path.splitext(item)[0]
                    
                    textures.append({
                        'name': texture_name,
                        'size': size_str,
                        'path': texture_path
                    })
            
            
            textures.sort(key=lambda x: x['name'].lower())
            
        except (OSError, IOError) as e:  pass
        return textures


    
    def _load_config(self, config_path="saves/config.json"):
        default_config = {
            "chunk_update_intreval": 1.0,
            "render_distance":       3,
            "min_fps_threshold":     5.0,
            "target_fps":            60.0
        }
        
        if not os.path.exists(config_path): return default_config
        

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:  config[key] = value
                return config
        except (json.JSONDecodeError, IOError, OSError):
            return default_config
    
    def _save_config(self, config_path="saves/config.json"):
        
        self.settings_config["chunk_update_intreval"] = self.settings_sliders[0]["value"]
        self.settings_config["render_distance"]  = int(self.settings_sliders[1]["value"])
        self.settings_config["target_fps"] = self.settings_sliders[2]["value"]
        

        
        if self.textures_current_selected is not None and 0 <= self.textures_current_selected < len(self.textures_list):
            self.settings_config["texture"] = self.textures_list[self.textures_current_selected]["name"]
        else:
            self.settings_config["texture"] = "none"
        try:
            os.makedirs("saves", exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.settings_config, f, indent=4)
        except (IOError, OSError) as e:
            raise valueError(f"Failed to save config: {e}")
    
    def load_texture(self, image_path, is_title=False):
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_data = np.array(img)
            
            if is_title:
                rgba_data = np.zeros((img.height, img.width, 4), dtype=np.uint8)
                rgba_data[:, :, :3] = img_data
                
                bg_color = np.array([184, 61, 186])
                mask = np.all(img_data == bg_color, axis=2)
                rgba_data[mask, 3]  = 0
                rgba_data[~mask, 3] = 255
                
                img_data  = rgba_data
                format_type = GL_RGBA
            else:
                format_type = GL_RGB
            
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            glTexImage2D(
                GL_TEXTURE_2D, 0, format_type, img.width, img.height, 0, 
                format_type, GL_UNSIGNED_BYTE, img_data
            )
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            return texture_id
            
        except Exception as e:  return None
    
    def _get_mouse_gl_coords(self):
        try:
            mouse_x, mouse_y = win32api.GetCursorPos()
            if self.terminal_hwnd:
                client_pt = win32gui.ScreenToClient(self.terminal_hwnd, (mouse_x, mouse_y))
                client_x, client_y = client_pt
                left, top, right, bottom = win32gui.GetClientRect(self.terminal_hwnd)
                client_w = right  - left
                client_h = bottom - top
                gl_mouse_x = int(client_x * self.gl_width / client_w)
                gl_mouse_y = int((client_h - client_y) * self.gl_height / client_h)
                return gl_mouse_x, gl_mouse_y
        except: pass

        return -1, -1




    
    def _find_window(self):
        hwnd = win32gui.GetForegroundWindow()
        self.terminal_hwnd = hwnd
        return True
    
    def _is_mouse_button_pressed(self):
        try: return (win32api.GetKeyState(0x01) & 0x8000) != 0
        except: return False



    
    def _init_color_system(self):
        if curses.COLORS >= 256:
            self.color_cache  = {}
            self.pair_cache   = {}
            self.next_pair_id = 1
            self.xterm_colors = self._precompute_full_xterm_colors()


        else:
            for i in range(min(8, curses.COLORS)):
                curses.init_pair(i + 1, i, curses.COLOR_BLACK)
    


    def _precompute_full_xterm_colors(self):
        colors = []
        
        standard_colors = [
            (0,   0,   0  ), (128, 0, 0  ), (0, 128, 0  ), (128, 128, 0  ),
            (0,   0,   128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0  ), (0, 255, 0  ), (255, 255, 0  ),
            (0,   0,   255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
        ]
        colors.extend(standard_colors)
        
        for r in range(6):
            for g in range(6):
                for b in range(6):                  # 50
                    r_val = 0 if r == 0 else 55 + r * 40
                    g_val = 0 if g == 0 else 55 + g * 40
                    b_val = 0 if b == 0 else 55 + b * 40
                    colors.append((r_val, g_val, b_val))
        
        for i in range(24):
            gray = 8 + i * 10
            colors.append((gray, gray, gray))
        
        return colors
    
    def rgb_to_color_index(self, r, g, b):
        if curses.COLORS < 256:  return 1
            
        key = (r, g, b)
        if key in self.color_cache:
            return self.color_cache[key]
        
        r_scaled = int(r * 255)
        g_scaled = int(g * 255)
        b_scaled = int(b * 255)
        
        best_idx = 0
        best_distance = float('inf')
        

        for i, (cr, cg, cb) in enumerate(self.xterm_colors):
            distance = (cr - r_scaled)**2 + (cg - g_scaled)**2 + (cb - b_scaled)**2
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        self.color_cache[key] = best_idx
        return best_idx
    




    def get_color_pair(self, fg_color, bg_color):
        if curses.COLORS < 256:  return 1
            
        pair_key = (fg_color, bg_color)
        if pair_key in self.pair_cache:
            return self.pair_cache[pair_key]
        
        max_pairs = min(curses.COLOR_PAIRS - 1, 32000)
        


        if self.next_pair_id > max_pairs:
            self.next_pair_id = 1
            self.pair_cache.clear()
        
        try:
            pair_id = self.next_pair_id
            curses.init_pair(pair_id, fg_color, bg_color)
            self.pair_cache[pair_key] = pair_id
            self.next_pair_id += 1
            return pair_id
        except curses.error:  return 1
    
    def render_image(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(0, 0)
            glTexCoord2f(1, 1); glVertex2f(self.gl_width, 0)
            glTexCoord2f(1, 0); glVertex2f(self.gl_width, self.gl_height)
            glTexCoord2f(0, 0); glVertex2f(0, self.gl_height)
            glEnd()
            
            glDisable(GL_TEXTURE_2D)
        
        if self.title_texture_id and self.active_menu is None:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.title_texture_id)
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(self.title_x,  self.title_y)
            glTexCoord2f(1, 1); glVertex2f(self.title_x + self.title_width, self.title_y)
            glTexCoord2f(1, 0); glVertex2f(self.title_x + self.title_width, self.title_y + self.title_height)
            glTexCoord2f(0, 0); glVertex2f(self.title_x,  self.title_y + self.title_height)
            glEnd()
            
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
        
        self.render_ui_overlay()
        pygame.display.flip()
    
    def update_ui(self):
        if not self.terminal_hwnd: self._find_window()
        
        self.mouse_x, self.mouse_y = self._get_mouse_gl_coords()


        
        if self.active_menu is not None:
            self._handle_menu_ui()
            return
        
        if self.mouse_x >= 0 and self.mouse_y >= 0:
            hovered_button = self.get_hovered_button(self.mouse_x, self.mouse_y)
            if hovered_button is not None:
                self.ui_selected_button = hovered_button
        
        key = self.stdscr.getch()
        if key == curses.KEY_UP:     self.ui_selected_button = (self.ui_selected_button - 1) % len(self.ui_buttons)
        elif key == curses.KEY_DOWN: self.ui_selected_button = (self.ui_selected_button + 1) % len(self.ui_buttons)
        elif key == ord('\n') or key == ord(' '): self._handle_button_click()
        

        current_mouse_state = self._is_mouse_button_pressed()
        
        if current_mouse_state and not self.last_mouse_state:
            if self.mouse_x >= 0 and self.mouse_y >= 0:
                hovered_button = self.get_hovered_button(self.mouse_x, self.mouse_y)
                if hovered_button is not None:
                    self.ui_selected_button = hovered_button
                    self._handle_button_click()
        
        self.last_mouse_state = current_mouse_state
    


    # -- menu handlers --
    # on width  1px = 1char
    # on height 2px = 1char
    def _handle_menu_ui(self):
        margin   = 8
        button_h = 5
        button_w = 35
        button_spacing = 3
        menu_buttons = ['Back', 'Continue']
        num_buttons = len(menu_buttons)
        total_buttons_width = button_w * num_buttons + button_spacing * (num_buttons - 1)

        # 1 char above terminal bottom
        buttons_y = 3 
        start_x = (self.gl_width - total_buttons_width) // 2

        hovered_widget = None
        hovered_button = None

        # layouts
        if self.active_menu == 'new_world':
            left  = margin
            right = self.gl_width - margin
            menu_bottom = buttons_y + button_h + 2
            menu_top = self.gl_height - 4
            widget_w = 60
            widget_h = 9
            spacing  = 5
            num_widgets = 4
            top_margin  = 16
            center_x = (left + right) // 2
            prev_selected_field = self.nw_selected_field
            for i in range(num_widgets):
                x = center_x - widget_w // 2
                y = menu_top - top_margin - i * (widget_h + spacing)
                if x <= self.mouse_x <= x + widget_w and y <= self.mouse_y <= y + widget_h:
                    hovered_widget = i
                    break


        elif self.active_menu == 'load_world':
            left  = margin
            right = self.gl_width - margin
            menu_bottom = buttons_y + button_h + 2
            menu_top = self.gl_height - 4
            widget_w = 60
            widget_h = 9
            spacing  = 5
            # show 3 worlds + 1 create_new_world widget = 4 total
            num_worlds  = min(3, self.lw_num_buttons - self.lw_scroll_offset)
            num_widgets = num_worlds + 1  # +1 for nw widget
            center_x    = (left + right) // 2
            top_margin  = 16
            
            
            # nw hover
            x = center_x - widget_w // 2
            y = menu_top - top_margin

            if x <= self.mouse_x <= x + widget_w and y <= self.mouse_y <= y + widget_h:
                hovered_widget = 0
            else:
                for i in range(num_worlds):
                    x = center_x - widget_w // 2
                    y = menu_top - top_margin - (i + 1) * (widget_h + spacing)
                    if x <= self.mouse_x <= x + widget_w and y <= self.mouse_y <= y + widget_h:
                        hovered_widget = i + 1
                        break
            
            # arrow button hover
            arrow_size   = 8
            arrow_x      = center_x + widget_w // 2 + 5
            # center with 1st and last elems
            up_arrow_y   = menu_top - top_margin - arrow_size // 2 + 2 - 14
            down_arrow_y = menu_top - top_margin - (num_widgets - 1) * (widget_h + spacing) - arrow_size // 2 + 5
            
            if (arrow_x <= self.mouse_x <= arrow_x + arrow_size and 
                up_arrow_y <= self.mouse_y <= up_arrow_y + arrow_size):
                self.lw_scroll_up_hovered = True
            else:
                self.lw_scroll_up_hovered = False
                
            if (arrow_x <= self.mouse_x <= arrow_x + arrow_size and 
                down_arrow_y <= self.mouse_y <= down_arrow_y + arrow_size):
                self.lw_scroll_down_hovered = True
            else:
                self.lw_scroll_down_hovered = False


                
        elif self.active_menu == 'settings':

            left  = margin
            right = self.gl_width - margin
            menu_bottom = buttons_y + button_h + 2
            menu_top = self.gl_height - 4
            widget_w = 60
            widget_h = 9
            spacing  = 5
            num_widgets = 4  # 3 sliders + 1 button
            center_x    = (left + right) // 2
            top_margin  = 16
            
            
            # hover
            for i in range(num_widgets):
                x = center_x - widget_w // 2
                y = menu_top - top_margin - i * (widget_h + spacing)
                if x <= self.mouse_x <= x + widget_w and y <= self.mouse_y <= y + widget_h:
                    hovered_widget = i
                    break


                    
        elif self.active_menu == 'textures':
            
            left  = margin
            right = self.gl_width - margin
            menu_bottom = buttons_y + button_h + 2
            menu_top = self.gl_height - 4
            widget_w = 60
            widget_h = 9
            spacing  = 5
            num_widgets = min(self.textures_visible_count, self.textures_num_buttons - self.textures_scroll_offset)
            center_x    = (left + right) // 2
            top_margin  = 16
            
            
            # texture button hover
            for i in range(num_widgets):
                x = center_x - widget_w // 2
                y = menu_top - top_margin - i * (widget_h + spacing)
                if x <= self.mouse_x <= x + widget_w and y <= self.mouse_y <= y + widget_h:
                    hovered_widget = i
                    break
            
            # arrow button hover
            arrow_size = 8
            arrow_x    = center_x + widget_w // 2 + 5
            # center with 1st and last elems
            up_arrow_y   = menu_top - top_margin - arrow_size // 2 + 2
            down_arrow_y = menu_top - top_margin - (num_widgets - 1) * (widget_h + spacing) - arrow_size // 2 + 5
            
            if (arrow_x <= self.mouse_x <= arrow_x + arrow_size and 
                up_arrow_y <= self.mouse_y <= up_arrow_y + arrow_size):
                self.textures_scroll_up_hovered = True
            else:
                self.textures_scroll_up_hovered = False
                
            if (arrow_x <= self.mouse_x <= arrow_x + arrow_size and 
                down_arrow_y <= self.mouse_y <= down_arrow_y + arrow_size):
                self.textures_scroll_down_hovered = True
            else:
                self.textures_scroll_down_hovered = False
        
        

        if self.mouse_x >= 0 and self.mouse_y >= 0:
            for i in range(num_buttons):
                x = start_x + i * (button_w + button_spacing)
                y = buttons_y
                if x <= self.mouse_x <= x + button_w and y <= self.mouse_y <= y + button_h:
                    hovered_button = i
                    break

        

        # stop listening to keys (editing ) when hovering another widget 
        if self.active_menu == 'new_world':
            if hovered_widget is not None:
                
                if self.nw_editing and hovered_widget != self.nw_selected_field:
                    with self._edit_lock:
                        if self._edit_field == 0:   self.nw_world_name = self._edit_buffer
                        elif self._edit_field == 1: self.nw_world_seed = self._edit_buffer
                        self._edit_buffer = ""
                        self._edit_field  = None
                    self.nw_editing = False
                self.nw_selected_field = hovered_widget

            if hovered_button is not None:
                self.menu_selected_button = hovered_button
                if self.nw_editing:
                    with self._edit_lock:
                        if self._edit_field == 0:   self.nw_world_name = self._edit_buffer
                        elif self._edit_field == 1: self.nw_world_seed = self._edit_buffer
                        self._edit_buffer = ""
                        self._edit_field  = None
                    self.nw_editing = False


        elif self.active_menu == 'load_world':
            if hovered_widget is not None: self.lw_selected_button   = hovered_widget
            if hovered_button is not None: self.menu_selected_button = hovered_button
        elif self.active_menu == 'settings':
            if hovered_widget is not None: self.settings_selected_widget = hovered_widget
            if hovered_button is not None: self.menu_selected_button     = hovered_button
        elif self.active_menu == 'textures':
            if hovered_widget is not None: self.textures_selected_button = hovered_widget
            if hovered_button is not None: self.menu_selected_button     = hovered_button
        else:
            if hovered_button is not None: self.menu_selected_button     = hovered_button

        

        key = self.stdscr.getch()
        if self.active_menu == 'new_world':
            if self.nw_editing:
                self.stdscr.nodelay(1)
                with self._edit_lock:
                    if self._edit_field == 0:   self.nw_world_name = self._edit_buffer
                    elif self._edit_field == 1: self.nw_world_seed = self._edit_buffer
            else: self.stdscr.nodelay(1)

        
        current_mouse_state = self._is_mouse_button_pressed()
        if self.active_menu == 'new_world':
            if current_mouse_state and not self.last_mouse_state:
                # widget clicked
                if hovered_widget is not None:
                    self.nw_selected_field = hovered_widget
                    # click toggle -> flip value
                    if hovered_widget == 2:
                        self.nw_generate_features = not self.nw_generate_features
                        self.nw_editing = False
                        with self._edit_lock:
                            self._edit_buffer = ""
                            self._edit_field  = None
                    elif hovered_widget == 3:
                        self.nw_generate_caves = not self.nw_generate_caves
                        self.nw_editing = False
                        with self._edit_lock:
                            self._edit_buffer = ""
                            self._edit_field  = None
                    # text fields -> begin edit listening
                    elif hovered_widget in (0, 1):
                        with self._edit_lock:
                            self._edit_field = hovered_widget
                            if hovered_widget   == 0: self._edit_buffer = self.nw_world_name
                            elif hovered_widget == 1: self._edit_buffer = self.nw_world_seed
                        self.nw_editing = True
                
                
                # button click
                if hovered_button is not None:
                    self.menu_selected_button = hovered_button
                    self._handle_menu_button_click()
                    self.nw_editing = False
                    with self._edit_lock:
                        self._edit_buffer = ""
                        self._edit_field  = None


        elif self.active_menu == 'load_world':
            if current_mouse_state and not self.last_mouse_state:
                if hovered_widget is not None:
                    self.lw_selected_button = hovered_widget

                    if hovered_widget == 0: self.active_menu = 'new_world'
                    else:
                        # set selected world
                        world_index = hovered_widget - 1 + self.lw_scroll_offset  # -1 because nw is at 0
                        if world_index < len(self.lw_worlds): self.lw_current_selected = world_index

                elif self.lw_scroll_up_hovered and self.lw_scroll_offset > 0:
                    self.lw_scroll_offset -= 1
                elif self.lw_scroll_down_hovered and self.lw_scroll_offset + 3 < self.lw_num_buttons:  # show 3 worlds at a time
                    self.lw_scroll_offset += 1

                if hovered_button is not None:
                    # is continue disabled
                    continue_disabled = (hovered_button == 1 and 
                        (self.lw_current_selected < 0 or len(self.lw_worlds) == 0)
                    )
                    if not continue_disabled:
                        self.menu_selected_button = hovered_button
                        self._handle_menu_button_click()


                        
        elif self.active_menu == 'settings':
            if current_mouse_state and not self.last_mouse_state:
                if hovered_widget is not None:

                    self.settings_selected_widget = hovered_widget
                    if hovered_widget == 3:  self.active_menu = 'textures'
                    else:  # slider
                        # begin drag
                        self.settings_dragging_slider  = True
                        self.settings_drag_start_x     = self.mouse_x
                        self.settings_drag_start_value = self.settings_sliders[hovered_widget]["value"]
                if hovered_button is not None:
                    self.menu_selected_button = hovered_button
                    self._handle_menu_button_click()
            
            # handle drag
            if self.settings_dragging_slider and current_mouse_state:
                if self.settings_selected_widget < 3:
                    slider  = self.settings_sliders[self.settings_selected_widget]
                    delta_x = self.mouse_x - self.settings_drag_start_x
                    # pixel movement -> value change
                    widget_w     = 60
                    value_range  = slider["max"] - slider["min"]
                    value_change = (delta_x / widget_w) * value_range
                    new_value    = self.settings_drag_start_value + value_change
                    slider["value"] = max(slider["min"], min(slider["max"], new_value))
            
            if not current_mouse_state:  self.settings_dragging_slider = False



        elif self.active_menu == 'textures':
            if current_mouse_state and not self.last_mouse_state:
                if hovered_widget is not None: # TODO select texture
                    self.textures_selected_button = hovered_widget
                    texture_index = hovered_widget + self.textures_scroll_offset 
                    if texture_index < len(self.textures_list):
                        self.textures_current_selected = texture_index
                        
                        self._save_config()

                elif self.textures_scroll_up_hovered and self.textures_scroll_offset > 0:
                    self.textures_scroll_offset -= 1
                elif (
                    self.textures_scroll_down_hovered and 
                    self.textures_scroll_offset + self.textures_visible_count < self.textures_num_buttons
                    ):
                    self.textures_scroll_offset += 1
                if hovered_button is not None:
                    self.menu_selected_button = hovered_button
                    self._handle_menu_button_click()
        
        else:
            if current_mouse_state and not self.last_mouse_state:
                if hovered_button is not None:
                    self.menu_selected_button = hovered_button
                    self._handle_menu_button_click()
        self.last_mouse_state = self._is_mouse_button_pressed()




    def _handle_button_click(self):
        # home buttons
        if   self.ui_selected_button == 0: self.running = False
        elif self.ui_selected_button == 1: self.active_menu = 'settings'
        elif self.ui_selected_button == 2: self.active_menu = 'load_world'
    
    def _handle_menu_button_click(self):
        # when back button pressed
        if self.menu_selected_button == 0:
            # reset nw settings
            seed = random.randint(10000, 99999)
            self.nw_world_name = f"world_{seed}"
            self.nw_world_seed = str(seed)
            self.nw_generate_features = True
            self.nw_generate_caves    = True
            with self._edit_lock:
                self._edit_buffer = ""
                self._edit_field  = None
            self.nw_editing  = False
            self.active_menu = None

        elif self.menu_selected_button == 1:
            if self.active_menu == 'settings':
                self._save_config()
                self.active_menu = None

            elif self.active_menu == 'load_world':
                self._launch_game()

            elif self.active_menu == 'new_world':
                world_name = self._create_new_world()
                if world_name: # create folder then launch
                    self._launch_game_with_world(world_name)
            else: self.active_menu = None
    



    def _create_new_world(self):
        """create a new world folder with settings"""
        
        # handle duplicates
        base_name = self.nw_world_name.strip()
        if not base_name:  base_name = "world"
        # world -> world(i)
        world_name = base_name
        counter    = 1

        while os.path.exists(os.path.join("saves", world_name)):
            world_name = f"{base_name}({counter})"
            counter += 1
        
        
        world_path = os.path.join("saves", world_name)
        try:  os.makedirs(world_path, exist_ok=True)
        except OSError as e:
            print(f"FAIL at World Creation: {e}")
            return None
        
        # data.json
        world_data = {
            "seed": int(self.nw_world_seed) if self.nw_world_seed.isdigit() else 9999,
            "generateFeatures":  self.nw_generate_features,
            "generateCaveWorms": self.nw_generate_caves
        }
        
        with open(os.path.join(world_path, "data.json"), 'w') as f:
            json.dump(world_data, f, indent=4)
        
        return world_name
    

    
    def _launch_game_with_world(self, world_name):
        """launch main with args"""
        
        
        texture_name = "basic"
        if self.textures_current_selected is not None and 0 <= self.textures_current_selected < len(self.textures_list):
            texture_name = self.textures_list[self.textures_current_selected]['name']
        
        
        cmd = [
            sys.executable, "main.py",
            "--chunk_update_intreval", str(self.settings_sliders[0]["value"]),
            "--render_distance",       str(int(self.settings_sliders[1]["value"])),
            "--min_fps_threshold",     str(self.settings_config.get("min_fps_threshold", 5.0)),
            "--target_fps",            str(self.settings_sliders[2]["value"]),
            "--texture", texture_name,
            "--world",   world_name
        ]
        self.cleanup()
        
        subprocess.run(cmd, check=True)
        sys.exit(0)
    
    def _launch_game(self):
        
        
        if self.lw_current_selected >= 0 and self.lw_current_selected < len(self.lw_worlds):
            world_name = self.lw_worlds[self.lw_current_selected]['name']
        else: return
        
        
        self._launch_game_with_world(world_name)
    


    def get_button_rects(self):
        total_height = len(self.ui_buttons) * self.ui_button_height + (len(self.ui_buttons) - 1) * self.ui_button_spacing
        title_bottom    = self.title_y
        available_space = title_bottom
        start_y  = (available_space - total_height) // 2
        center_x = self.gl_width // 2
        
        rects = []
        for i in range(len(self.ui_buttons)):
            x = center_x - self.ui_button_width // 2
            y = start_y + i * (self.ui_button_height + self.ui_button_spacing)
            rects.append((x, y, self.ui_button_width, self.ui_button_height))
        return rects
    
    def get_hovered_button(self, mouse_x, mouse_y):
        rects = self.get_button_rects()
        for i, (x, y, w, h) in enumerate(rects):
            if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                return i
        return None



    # -- UI and rendering --

    def render_ui_overlay(self):
        glDisable(GL_TEXTURE_2D)
        
        if self.active_menu is not None:
            self._render_menu()
            return
        
        total_height = len(self.ui_buttons) * self.ui_button_height + (len(self.ui_buttons) - 1) * self.ui_button_spacing
        title_bottom = self.title_y
        available_space = title_bottom
        start_y  = (available_space - total_height) // 2
        center_x = self.gl_width // 2
        
        for i, button_text in enumerate(self.ui_buttons):
            x = center_x - self.ui_button_width // 2
            y = start_y + i * (self.ui_button_height + self.ui_button_spacing)
            
            if i == self.ui_selected_button:
                glColor3f(*self.ui_button_selected_color)
            else:
                glColor3f(*self.ui_button_color)
            
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + self.ui_button_width, y)
            glVertex2f(x + self.ui_button_width, y + self.ui_button_height)
            glVertex2f(x, y + self.ui_button_height)
            glEnd()
            
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + self.ui_button_width, y)
            glVertex2f(x + self.ui_button_width, y + self.ui_button_height)
            glVertex2f(x, y + self.ui_button_height)
            glEnd()
        
        glColor3f(1.0, 1.0, 1.0)
    
    def _render_menu(self):
        margin   = 8
        button_h = 5
        button_w = 35
        button_spacing = 3
        menu_buttons   = ['Back', 'Continue']
        num_buttons    = len(menu_buttons)
        total_buttons_width = button_w * num_buttons + button_spacing * (num_buttons - 1)

        # place menu rectangle with margin from top and buttons
        left  = margin
        right = self.gl_width - margin
        buttons_y   = 3  # opengl y coord for button bottoms
        menu_bottom = buttons_y + button_h + 2  # 2px margin above
        menu_top    = self.gl_height - 4        # 2px margin below
        menu_height = menu_top - menu_bottom

        glColor3f(0.12, 0.12, 0.12)
        glBegin(GL_QUADS)
        glVertex2f(left, menu_bottom)
        glVertex2f(right, menu_bottom)
        glVertex2f(right, menu_top)
        glVertex2f(left, menu_top)
        glEnd()

        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINE_LOOP)
        glVertex2f(left+1, menu_bottom)
        glVertex2f(right, menu_bottom)
        glVertex2f(right, menu_top)
        glVertex2f(left, menu_top)
        glEnd()

        
        if self.active_menu == 'new_world':  self._render_new_world_menu_widgets(left, right, menu_bottom, menu_top)
        if self.active_menu == 'load_world': self._render_load_world_menu_widgets(left, right, menu_bottom, menu_top)
        if self.active_menu == 'settings':   self._render_settings_menu_widgets(left, right, menu_bottom, menu_top)
        if self.active_menu == 'textures':   self._render_textures_menu_widgets(left, right, menu_bottom, menu_top)

        center_x = self.gl_width // 2
        start_x  = center_x - (total_buttons_width // 2)

        for i, button_text in enumerate(menu_buttons):
            x = start_x + i * (button_w + button_spacing)
            y = buttons_y

            
            continue_disabled = (
                self.active_menu == 'load_world' and 
                i == 1 and  # continue button
                (self.lw_current_selected < 0 or len(self.lw_worlds) == 0)
            )

            if continue_disabled:                glColor3f(0.3, 0.3, 0.3)
            elif i == self.menu_selected_button: glColor3f(0.7, 0.7, 0.7)
            else:                                glColor3f(0.5, 0.5, 0.5)

            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + button_w, y)
            glVertex2f(x + button_w, y + button_h)
            glVertex2f(x, y + button_h)
            glEnd()

            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + button_w, y)
            glVertex2f(x + button_w, y + button_h)
            glVertex2f(x, y + button_h)
            glEnd()

        glColor3f(1.0, 1.0, 1.0)



    def _render_new_world_menu_widgets(self, left, right, menu_bottom, menu_top):
        # 4 widgets starting from the top with margin in between
        widget_w    = 60
        widget_h    = 9
        spacing     = 5
        num_widgets = 4
        top_margin  = 14
        center_x    = (left + right) // 2
        labels = ["World Name", "World Seed", "Generate Features", "Generate Caves"]
        values = [self.nw_world_name, self.nw_world_seed, self.nw_generate_features, self.nw_generate_caves]
        for i in range(num_widgets):
            x = center_x - widget_w // 2
            y = menu_top - top_margin - i * (widget_h + spacing)
            
            if self.nw_selected_field == i:  glColor3f(0.7, 0.7, 0.7) # highlight
            else:                            glColor3f(0.5, 0.5, 0.5)


            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
            # border
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
        # TODO mouse hover /click detection



    def _render_load_world_menu_widgets(self, left, right, menu_bottom, menu_top):
        widget_w = 60
        widget_h = 9
        spacing  = 5

        # 3 lw elems + 1 nw widget
        num_worlds = min(3, self.lw_num_buttons - self.lw_scroll_offset)
        center_x   = (left + right) // 2
        top_margin = 14
        
        
        
        x = center_x - widget_w // 2
        y = menu_top - top_margin
        
        if self.lw_selected_button == 0:  glColor3f(0.7, 0.7, 0.7)
        else:                             glColor3f(0.5, 0.5, 0.5)
        
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + widget_w, y)
        glVertex2f(x + widget_w, y + widget_h)
        glVertex2f(x, y + widget_h)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x+1, y)
        glVertex2f(x + widget_w, y)
        glVertex2f(x + widget_w, y + widget_h)
        glVertex2f(x, y + widget_h)
        glEnd()
        

        # lw elems
        for i in range(num_worlds):
            x = center_x - widget_w // 2
            # y = menu_top - top_margin - (i) * (widget_h + spacing)
            y = menu_top - top_margin - (i + 1) * (widget_h + spacing)
            world_index = i + self.lw_scroll_offset
            
            if world_index == self.lw_current_selected: glColor3f(0.6, 0.8, 0.6)
            elif self.lw_selected_button == i + 1:      glColor3f(0.7, 0.7, 0.7)
            else:                                       glColor3f(0.5, 0.5, 0.5)
            
            
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
        
        # -- arrows --
        arrow_size   = 7
        arrow_x      = center_x + widget_w // 2 + 5 
        # center with 1st and last lw
        up_arrow_y   = menu_top - top_margin - (widget_h + spacing) - arrow_size // 2 + 4
        down_arrow_y = menu_top - top_margin - (num_worlds + 1) * (widget_h + spacing) - arrow_size // 2 + 4 + 14
        
        # up arrow
        if self.lw_scroll_offset > 0:
            if self.lw_scroll_up_hovered: glColor3f(0.7, 0.7, 0.7)
            else:                         glColor3f(0.5, 0.5, 0.5)
        else:                             glColor3f(0.3, 0.3, 0.3) # disabled
        
        glBegin(GL_QUADS)
        glVertex2f(arrow_x, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y + arrow_size)
        glVertex2f(arrow_x, up_arrow_y + arrow_size)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(arrow_x+1, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y + arrow_size)
        glVertex2f(arrow_x, up_arrow_y + arrow_size)
        glEnd()





        
        # down arrow
        if self.lw_scroll_offset + 3 < self.lw_num_buttons:
            if self.lw_scroll_down_hovered: glColor3f(0.7, 0.7, 0.7)
            else:                           glColor3f(0.5, 0.5, 0.5)
        else:                               glColor3f(0.3, 0.3, 0.3)  # disabled
        


        glBegin(GL_QUADS)
        glVertex2f(arrow_x, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y + arrow_size)
        glVertex2f(arrow_x, down_arrow_y + arrow_size)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(arrow_x+1, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y + arrow_size)
        glVertex2f(arrow_x, down_arrow_y + arrow_size)
        glEnd()



    def _render_textures_menu_widgets(self, left, right, menu_bottom, menu_top):
        widget_w = 60
        widget_h = 9
        spacing  = 5
        num_widgets = min(self.textures_visible_count, self.textures_num_buttons - self.textures_scroll_offset)
        center_x    = (left + right) // 2
        top_margin  = 14
        
        
        # texture list
        for i in range(num_widgets):
            x = center_x - widget_w // 2
            y = menu_top - top_margin - i * (widget_h + spacing)
            texture_index = i + self.textures_scroll_offset
            
            if texture_index == self.textures_current_selected: glColor3f(0.6, 0.8, 0.6)
            elif self.textures_selected_button == i:            glColor3f(0.7, 0.7, 0.7)
            else:                                               glColor3f(0.5, 0.5, 0.5)


            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()

            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
        
        # -- arrows --
        arrow_size   = 7
        arrow_x      = center_x + widget_w // 2 + 5
        # center with 1st and last lw
        up_arrow_y   = menu_top - top_margin - arrow_size // 2 + 4
        down_arrow_y = menu_top - top_margin - (num_widgets - 1) * (widget_h + spacing) - arrow_size // 2 + 4
        
        # up arrow
        if self.textures_scroll_offset > 0:
            if self.textures_scroll_up_hovered: glColor3f(0.7, 0.7, 0.7)
            else:                               glColor3f(0.5, 0.5, 0.5)
        else:                                   glColor3f(0.3, 0.3, 0.3)  # disableed
        
        glBegin(GL_QUADS)
        glVertex2f(arrow_x, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y + arrow_size)
        glVertex2f(arrow_x, up_arrow_y + arrow_size)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(arrow_x+1, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y)
        glVertex2f(arrow_x + arrow_size, up_arrow_y + arrow_size)
        glVertex2f(arrow_x, up_arrow_y + arrow_size)
        glEnd()
        
        # down arrow
        if self.textures_scroll_offset + self.textures_visible_count < self.textures_num_buttons:
            if self.textures_scroll_down_hovered: glColor3f(0.7, 0.7, 0.7)
            else:                                 glColor3f(0.5, 0.5, 0.5)
        else:                                     glColor3f(0.3, 0.3, 0.3)  # disabled
        
        glBegin(GL_QUADS)
        glVertex2f(arrow_x, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y + arrow_size)
        glVertex2f(arrow_x, down_arrow_y + arrow_size)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(arrow_x+1, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y)
        glVertex2f(arrow_x + arrow_size, down_arrow_y + arrow_size)
        glVertex2f(arrow_x, down_arrow_y + arrow_size)
        glEnd()





    # -- text overlay --

    def _draw_button_text(self):
        # top line thingy
        if hasattr(self, 'frame_count') and hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps   = self.frame_count / elapsed
                stats = f"Version 3.0.1 | FPS: {fps:.1f}"
                name  = "https://github.com/hexanitrohexaazaisowurtzitane"
                # you are free to modify the project as you wish, but please do not remove the credits
                self.stdscr.addstr(0, 0, stats, curses.A_REVERSE)
                self.stdscr.addstr(0, self.term_width - len(name), name, curses.A_REVERSE)
        
        if self.active_menu is not None:
            self._draw_menu_text()
            return
        


        total_height = len(self.ui_buttons) * self.ui_button_height + (len(self.ui_buttons) - 1) * self.ui_button_spacing
        title_bottom    = self.title_y
        available_space = title_bottom
        start_y         = (available_space - total_height) // 2
        
        for i, button_text in enumerate(self.ui_buttons):
            y = start_y + i * (self.ui_button_height + self.ui_button_spacing)
            
            button_top_term    = int((self.term_height * (self.gl_height - y)) // self.gl_height)
            button_bottom_term = int((self.term_height * (self.gl_height - (y + self.ui_button_height))) // self.gl_height)
            term_x             = max(0, (self.term_width - len(button_text)) // 2)
            term_y             = (button_top_term + button_bottom_term) // 2
            
            
            if 0 <= term_y < self.term_height and 0 <= term_x < self.term_width:
                try:
                    if i == self.ui_selected_button: self.stdscr.addstr(term_y, term_x, button_text, curses.A_REVERSE)
                    else:                            self.stdscr.addstr(term_y, term_x, button_text)
                except curses.error: pass
    


    def _draw_menu_text(self):
        menu_titles = {
            'settings':   'Settings Menu',
            'load_world': 'Load World Menu',
            'new_world':  'New World Menu',
            'textures':   'Textures Menu',
        }
        title = menu_titles.get(self.active_menu, 'Menu')

        margin   = 8
        button_h = 5
        button_w = 35
        button_spacing = 3
        menu_buttons   = ['Back', 'Continue']
        num_buttons    = len(menu_buttons)
        total_buttons_width = button_w * num_buttons + button_spacing * (num_buttons - 1)

        
        menu_top  = self.gl_height - 2
        menu_bottom = 2 + button_h + 2
        # opengl y coord -> rows
        menu_top_term    = int((self.term_height * (self.gl_height - menu_top))    // self.gl_height)
        menu_bottom_term = int((self.term_height * (self.gl_height - menu_bottom)) // self.gl_height)
        term_x = max(0, (self.term_width - len(title)) // 2)
        term_y = menu_top_term + 1
        

        if 0 <= term_y < self.term_height and 0 <= term_x < self.term_width:
            try: self.stdscr.addstr(term_y, term_x, title, curses.A_BOLD)
            except curses.error: pass




        # menu buttons
        button_term_y = self.term_height - 2
        start_x_gl    = (self.gl_width // 2) - (total_buttons_width // 2)

        for i, button_text in enumerate(menu_buttons):
            x_gl = start_x_gl + i * (button_w + button_spacing)
            x_gl_center = x_gl + button_w // 2
            term_x = int((self.term_width * x_gl_center) // self.gl_width) - (len(button_text) // 2)

            if 0 <= button_term_y < self.term_height and 0 <= term_x < self.term_width:
                try:
                    continue_disabled = (
                        self.active_menu == 'load_world' and 
                        i == 1 and  # continue button
                        (self.lw_current_selected < 0 or len(self.lw_worlds) == 0)
                    )
                    
                    if continue_disabled:                self.stdscr.addstr(button_term_y-1, term_x, button_text, curses.A_DIM)
                    elif i == self.menu_selected_button: self.stdscr.addstr(button_term_y-1, term_x, button_text, curses.A_REVERSE)
                    else:                                self.stdscr.addstr(button_term_y-1, term_x, button_text)
                except curses.error: pass
    
    

        
        if self.active_menu == 'new_world':  self._draw_new_world_menu_text(menu_top_term, menu_bottom_term)
        if self.active_menu == 'load_world': self._draw_load_world_menu_text(menu_top_term, menu_bottom_term)
        if self.active_menu == 'settings':   self._draw_settings_menu_text(menu_top_term, menu_bottom_term)
        if self.active_menu == 'textures':   self._draw_textures_menu_text(menu_top_term, menu_bottom_term)



    def _draw_new_world_menu_text(self, menu_top_term, menu_bottom_term):
        # 4 widget drop from top
        widget_height_rows = 5  
        spacing_rows       = 2 
        num_widgets        = 4
        top_margin_rows    = 4
        labels = [
            "World Name" + (" : editing" if self.nw_editing and self.nw_selected_field == 0 else ""),
            "World Seed" + (" : editing" if self.nw_editing and self.nw_selected_field == 1 else ""),
            "Generate Features",
            "Generate Caves"
        ]
        values = [
            self.nw_world_name,
            self.nw_world_seed,
            "ON" if self.nw_generate_features else "OFF",
            "ON" if self.nw_generate_caves    else "OFF"
        ]


        widget_width_chars = 30
        for i in range(num_widgets):
            y = menu_top_term + top_margin_rows + i * (widget_height_rows + spacing_rows)
            x = max(0, (self.term_width - widget_width_chars) // 2)
            label_x = (self.term_width - widget_width_chars - 24) // 2
            # label on left
            try:
                if self.nw_selected_field == i: self.stdscr.addstr(y-1, label_x, f"> {labels[i]}  ", curses.A_REVERSE)
                else:                           self.stdscr.addstr(y-1, label_x, f"  {labels[i]}  ")
            except curses.error: pass
            
            # value on center
            value = values[i]
            value_x = x + (widget_width_chars - len(str(value))) // 2
            try:
                if self.nw_selected_field == i: self.stdscr.addstr(y+2, value_x, str(value), curses.A_REVERSE)
                else:                           self.stdscr.addstr(y+2, value_x, str(value))
            except curses.error: pass
    



    def _draw_load_world_menu_text(self, menu_top_term, menu_bottom_term):
        widget_height_rows = 5
        spacing_rows       = 2
        # 3 lw + 1 nw
        num_worlds         = min(3, self.lw_num_buttons - self.lw_scroll_offset)
        top_margin_rows    = 4
        widget_width_chars = 30

        
        rect_left = max(0, (self.term_width - widget_width_chars) // 2) - 13
        rect_left = max(0, rect_left)
        rect_right = rect_left + widget_width_chars + 26  # 13*2 pad

        # nw wfiget
        y = menu_top_term + top_margin_rows
        x = rect_left
        
        try:
            if self.lw_selected_button == 0:
                self.stdscr.addstr(y + 1, x, "Create New World",        curses.A_REVERSE)
                self.stdscr.addstr(y + 2, x, "Start a fresh adventure", curses.A_REVERSE)
                
                create_text = "< Create >"
                create_x    = rect_right - len(create_text)
                self.stdscr.addstr(y + 2, create_x, create_text, curses.A_REVERSE)
            else:
                self.stdscr.addstr(y + 1, x, "Create New World")
                self.stdscr.addstr(y + 2, x, "Start a fresh adventure")
        
        except curses.error:  pass

        # lw widgets
        if self.lw_num_buttons == 0:
            # TODO actallu handle this
            y = menu_top_term + top_margin_rows + (widget_height_rows + spacing_rows)
            x = max(0, (self.term_width - 20) // 2)
            try: self.stdscr.addstr(y, x, "No worlds found")
            except curses.error: pass
            return

        for i in range(num_worlds):
            #   menu_top_term + top_margin_rows + i * (widget_height_rows + spacing_rows) 
            y = menu_top_term + top_margin_rows + (i + 1) * (widget_height_rows + spacing_rows) 
            x = rect_left

            world_data = self.lw_worlds[i + self.lw_scroll_offset]
            world_name = world_data['name']
            world_seed = world_data['seed']
            world_size = world_data['size']

            try:
                world_index = i + self.lw_scroll_offset
                if world_index == self.lw_current_selected:
                    # highlight
                    self.stdscr.addstr(y + 1, x, "World: " + world_name, curses.A_BOLD)
                    self.stdscr.addstr(y + 2, x, "Seed:  " + world_seed, curses.A_BOLD)
                    self.stdscr.addstr(y + 3, x, "Size:  " + world_size, curses.A_BOLD)
                    
                    current_text = "< Current >"
                    current_x = rect_right - len(current_text)
                    if current_x > x + len("Seed: " + world_seed) + 1:
                        self.stdscr.addstr(y + 2, current_x, current_text, curses.A_BOLD)
                
                # elif self.lw_selected_button == i: 
                elif self.lw_selected_button == i + 1: 
                    self.stdscr.addstr(y + 1, x, "World: " + world_name, curses.A_REVERSE)
                    self.stdscr.addstr(y + 2, x, "Seed:  " + world_seed, curses.A_REVERSE)
                    self.stdscr.addstr(y + 3, x, "Size:  " + world_size, curses.A_REVERSE)
                    
                    play_text = "< Play >"
                    play_x = rect_right - len(play_text)
                    if play_x > x + len("Seed: " + world_seed) + 1:
                        self.stdscr.addstr(y + 2, play_x, play_text, curses.A_REVERSE)
                else:
                    self.stdscr.addstr(y + 1, x, "World: " + world_name)
                    self.stdscr.addstr(y + 2, x, "Seed:  " + world_seed)
                    self.stdscr.addstr(y + 3, x, "Size:  " + world_size)
            
            
            except curses.error: pass
        
        # arrow labels
        arrow_x_term      = int((self.term_width * (self.gl_width // 2 + 32)) // self.gl_width) + 7
        up_arrow_y_term   = menu_top_term + top_margin_rows + (widget_height_rows + spacing_rows) + 2  # center with 1st lw
        down_arrow_y_term = menu_top_term + top_margin_rows + (num_worlds + 1) * (widget_height_rows + spacing_rows) + 2  # last lw
        try:
            if self.lw_scroll_offset > 0:
                if self.lw_scroll_up_hovered: self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲", curses.A_REVERSE)
                else:                         self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲")
            else:                             self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲", curses.A_DIM)
        except curses.error: pass
            
        try:
            if self.lw_scroll_offset + 3 < self.lw_num_buttons:
                if self.lw_scroll_down_hovered: self.stdscr.addstr(down_arrow_y_term-7, arrow_x_term, "▼", curses.A_REVERSE)
                else:                           self.stdscr.addstr(down_arrow_y_term-7, arrow_x_term, "▼")
            else:                               self.stdscr.addstr(down_arrow_y_term-7, arrow_x_term, "▼", curses.A_DIM)
        except curses.error: pass

    def _draw_settings_menu_text(self, menu_top_term, menu_bottom_term):
        widget_height_rows = 5
        spacing_rows       = 2
        num_widgets        = 4  # 3 sliders + 1 button
        top_margin_rows    = 4
        widget_width_chars = 30
        
        # sliders
        for i in range(3):
            y = menu_top_term + top_margin_rows + i * (widget_height_rows + spacing_rows)
            x = max(0, (self.term_width - widget_width_chars) // 2)
            label_x = (self.term_width - widget_width_chars - 24) // 2
            
            slider = self.settings_sliders[i]
            label = slider["name"]

            # steps
            if slider["step"] == 1:  value = f"{int(slider['value'])}"
            else:                    value = f"{slider['value']:.1f}"
            
            try:
                if self.settings_selected_widget == i: self.stdscr.addstr(y-1, label_x, f"> {label}  ", curses.A_REVERSE)
                else:                                  self.stdscr.addstr(y-1, label_x, f"  {label}  ")
            except curses.error: pass
            

            # value in centr
            value_x = x + (widget_width_chars - len(value)) // 2
            try:
                if self.settings_selected_widget == i: self.stdscr.addstr(y+1, value_x, value, curses.A_REVERSE)
                else:                                  self.stdscr.addstr(y+1, value_x, value)
            except curses.error: pass
        
        # textures
        y = menu_top_term + top_margin_rows + 3 * (widget_height_rows + spacing_rows)
        x = max(0, (self.term_width - widget_width_chars) // 2)
        
        textures_label = "Textures"
        label_x = x + (widget_width_chars - len(textures_label)) // 2
        
        try:
            if self.settings_selected_widget == 3: self.stdscr.addstr(y+2, label_x, textures_label, curses.A_REVERSE)
            else:                                  self.stdscr.addstr(y+2, label_x, textures_label)
        except curses.error: pass




    def _draw_textures_menu_text(self, menu_top_term, menu_bottom_term):
        widget_height_rows = 5
        spacing_rows       = 2
        num_widgets = min(self.textures_visible_count, self.textures_num_buttons - self.textures_scroll_offset)
        top_margin_rows    = 4
        widget_width_chars = 30

        
        rect_left = max(0, (self.term_width - widget_width_chars) // 2) - 13
        rect_left = max(0, rect_left)
        rect_right = rect_left + widget_width_chars + 26  # 13*2 pad




        if self.textures_num_buttons == 0:
            y = menu_top_term + top_margin_rows
            x = max(0, (self.term_width - 20) // 2)
            self.stdscr.addstr(y, x, "No textures found")
            return

        for i in range(num_widgets):
            y = menu_top_term + top_margin_rows + i * (widget_height_rows + spacing_rows)
            x = rect_left

            texture_data = self.textures_list[i + self.textures_scroll_offset]
            texture_name = texture_data['name']
            texture_size = texture_data['size']

            try:
                texture_index = i + self.textures_scroll_offset
                if texture_index == self.textures_current_selected:
                    
                    self.stdscr.addstr(y + 1, x, "Texture: " + texture_name, curses.A_BOLD)
                    self.stdscr.addstr(y + 2, x, "Size: "    + texture_size, curses.A_BOLD)
                    
                    current_text = "< Current >"
                    current_x = rect_right - len(current_text)
                    if current_x > x + len("Size: " + texture_size) + 1:
                        self.stdscr.addstr(y + 2, current_x, current_text, curses.A_BOLD)


                elif self.textures_selected_button == i:
                    self.stdscr.addstr(y + 1, x, "Texture: " + texture_name, curses.A_REVERSE)
                    self.stdscr.addstr(y + 2, x, "Size: "    + texture_size, curses.A_REVERSE)
                    
                    select_text = "< Select >"
                    select_x = rect_right - len(select_text)
                    if select_x > x + len("Size: " + texture_size) + 1:
                        self.stdscr.addstr(y + 2, select_x, select_text, curses.A_REVERSE)
                else:
                    self.stdscr.addstr(y + 1, x, "Texture: " + texture_name)
                    self.stdscr.addstr(y + 2, x, "Size: "    + texture_size)
            except curses.error:
                pass
        
        # arrow labels
        arrow_x_term      = int((self.term_width * (self.gl_width // 2 + 32)) // self.gl_width) + 7
        up_arrow_y_term   = menu_top_term + top_margin_rows + 2 
        down_arrow_y_term = menu_top_term + top_margin_rows + (num_widgets - 1) * (widget_height_rows + spacing_rows) + 2
        
        try:
            if self.textures_scroll_offset > 0:
                if self.textures_scroll_up_hovered: self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲", curses.A_REVERSE)
                else:                               self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲")
            else:                                   self.stdscr.addstr(up_arrow_y_term, arrow_x_term, "▲", curses.A_DIM)
        except curses.error: pass
            
        try:
            if self.textures_scroll_offset + self.textures_visible_count < self.textures_num_buttons:
                if self.textures_scroll_down_hovered: self.stdscr.addstr(down_arrow_y_term, arrow_x_term, "▼", curses.A_REVERSE)
                else:                                 self.stdscr.addstr(down_arrow_y_term, arrow_x_term, "▼")
            else:                                     self.stdscr.addstr(down_arrow_y_term, arrow_x_term, "▼", curses.A_DIM)
        except curses.error:  pass

    def _render_settings_menu_widgets(self, left, right, menu_bottom, menu_top):
        center_x = (left + right) // 2
        widget_w = 60
        widget_h = 9
        spacing  = 5
        num_widgets = 4  # 3 sliders + 1 button
        top_margin  = 14
        
        
        # sliders
        for i in range(3):
            x = center_x - widget_w // 2
            y = menu_top - top_margin - i * (widget_h + spacing)
            
            if self.settings_selected_widget == i: glColor3f(0.7, 0.7, 0.7)
            else:                                  glColor3f(0.5, 0.5, 0.5)

            
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
            
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x+1, y)
            glVertex2f(x + widget_w, y)
            glVertex2f(x + widget_w, y + widget_h)
            glVertex2f(x, y + widget_h)
            glEnd()
            

            # slider track
            track_x = x + 5
            track_y = y + widget_h // 2
            track_w = widget_w - 10
            track_h = 2
            
            glColor3f(0.3, 0.3, 0.3)
            glBegin(GL_QUADS)
            glVertex2f(track_x, track_y)
            glVertex2f(track_x + track_w, track_y)
            glVertex2f(track_x + track_w, track_y + track_h)
            glVertex2f(track_x, track_y + track_h)
            glEnd()
            
            # slider handle
            slider = self.settings_sliders[i]
            value_ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
            handle_x = track_x + value_ratio * track_w - 2
            handle_y = track_y - 1
            handle_w = 4
            handle_h = 4
            
            glColor3f(0.8, 0.8, 0.8)
            glBegin(GL_QUADS)
            glVertex2f(handle_x, handle_y)
            glVertex2f(handle_x + handle_w, handle_y)
            glVertex2f(handle_x + handle_w, handle_y + handle_h)
            glVertex2f(handle_x, handle_y + handle_h)
            glEnd()
        


        # textures button
        x = center_x - widget_w // 2
        y = menu_top - top_margin - 3 * (widget_h + spacing)
        
        if self.settings_selected_widget == 3: glColor3f(0.7, 0.7, 0.7)
        else:                                  glColor3f(0.5, 0.5, 0.5)
        
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + widget_w, y)
        glVertex2f(x + widget_w, y + widget_h)
        glVertex2f(x, y + widget_h)
        glEnd()
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x+1, y)
        glVertex2f(x + widget_w, y)
        glVertex2f(x + widget_w, y + widget_h)
        glVertex2f(x, y + widget_h)
        glEnd()




    # -- render to terminal --

    def render_to_buffer(self):
        glReadBuffer(GL_BACK)
        pixel_data = glReadPixels(0, 0, self.gl_width, self.gl_height, GL_RGB, GL_UNSIGNED_BYTE)
        self.pixel_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(self.gl_height, self.gl_width, 3)
        self.pixel_array = np.flipud(self.pixel_array)
    
    def display_to_terminal(self):
        for y in range(0, min(self.term_height, self.gl_height // 2)):
            y2 = y * 2


            if y2 + 1 >= self.gl_height: break
            
            top_row = self.pixel_array[y2, :min(self.term_width, self.gl_width)]     / 255.0
            bot_row = self.pixel_array[y2 + 1, :min(self.term_width, self.gl_width)] / 255.0
            
            top_colors = [self.rgb_to_color_index(r, g, b) for r, g, b in top_row]
            bot_colors = [self.rgb_to_color_index(r, g, b) for r, g, b in bot_row]
            
            for x in range(min(self.term_width, len(top_colors))):
                try:
                    pair_id = self.get_color_pair(bot_colors[x], top_colors[x])
                    self.stdscr.addstr(y, x, '▄', curses.color_pair(pair_id))

                except (curses.error, IndexError): pass
    
    def run(self):
        self.frame_count = 0
        self.start_time = time.time()
        
        while self.running:
            key = self.stdscr.getch()
            if key == ord('q') or key == 27: break
            
            self.update_ui()
            self.render_image()
            self.render_to_buffer()
            self.display_to_terminal()
            self._draw_button_text()
            self.stdscr.refresh()
            
            self.frame_count += 1
            time.sleep(0.016)
    
    def cleanup(self):
        if self.texture_id:       glDeleteTextures([self.texture_id])
        if self.title_texture_id: glDeleteTextures([self.title_texture_id])
        curses.endwin()
        pygame.quit()




    def _keyboard_listener(self):
        while True:
            if self.nw_editing:
                event = keyboard.read_event(suppress=False)
                if event.event_type == keyboard.KEY_DOWN:
                    with self._edit_lock:
                        if self._edit_field == 0:
                            # name: allows printable chars and backspace
                            if event.name == 'backspace':                           self._edit_buffer = self._edit_buffer[:-1]
                            elif len(event.name) == 1 and event.name.isprintable(): self._edit_buffer += event.name
                        elif self._edit_field == 1:
                            # seeed: allows digits and backspace
                            if event.name == 'backspace': self._edit_buffer = self._edit_buffer[:-1]
                            elif event.name.isdigit():    self._edit_buffer += event.name
            
            else:  keyboard.read_event(suppress=False)  # ignore when  not editing


# NOTE you can change these images, however beware of color limitations.
# for images with too many or too vivid colors, the render will break.
# i'll try to mediate this in the future, but for now keep these as simple as possible! 
renderer = ImageTerminalRenderer("background.png", "title.png", title_top_padding=2)
renderer.run()

