import numpy as np
import time
import curses

class PlayerController:
    # Beware: This class is now mostly used for physics calculations only,
    #         A lot of stuff was moved to the main controller class
    def __init__(self, chunk_manager):
        self.chunk_manager = chunk_manager
        
        self.position = np.array([0.0, 40.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.moving_forward  = False
        self.moving_backward = False
        self.moving_left     = False
        self.moving_right    = False
        self.moving_up       = False
        self.moving_down     = False
        self.jumping         = False


        # collision
        self.player_height = 1.8    # in blocks
        self.player_width  = 0.6    # in blocks
        self.feet_offset   = 0.05   # offset to detect ground below feet
        
        # constants
        self.gravity       = -20.0  # acceleration due to gravity
        self.jump_force    = 8.0    # initial velocity when jumping
        self.friction      = 0.8    
        self.air_friction  = 0.98   
        self.terminal_velocity = -30.0  # max falling speed
        
        # Movement properties
        self.movement_speed = 5.0  # basic move (blocks/sec)
        self.flight_speed   = 10.0 
        self.is_flying = False     
        self.on_ground = False     
        self.can_jump  = False     
        

        self.input_states = {
            'forward': False,  'backward': False,  'left': False,
            'right':   False,  'up':       False,  'down': False,
            'jump':    False,
        }
        self.last_jump_time = 0.0  
        self.jump_cooldown  = 0.3 
        
        
        self.collision_buffer = 0.01   #  prevent fall through
        
        # simulation
        self.physics_step_size = 0.02  # 20ms steps
        self.max_physics_steps = 5
    
    
    def toggle_flight_mode(self):
        self.is_flying  = not self.is_flying
        if self.is_flying: self.velocity[1] = 0.0
    
    
    def get_surrounding_blocks(self, position):
        pos_x, pos_y, pos_z = position
        
        
        check_radius = int(self.player_width  * 2.0) + 1 
        check_height = int(self.player_height * 2.0) + 1
        

        blocks = {}
        
        # get blocks from chunk manager
        min_x, min_y, min_z = int(pos_x - check_radius), int(pos_y - 2),            int(pos_z - check_radius)
        max_x, max_y, max_z = int(pos_x + check_radius), int(pos_y + check_height), int(pos_z + check_radius)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    
                    # skip air : position -> -1 if not found
                    chunk_x, chunk_z = self.chunk_manager.get_chunk_coords_for_position((x, 0, z))
                    chunk_pos = (chunk_x, chunk_z)
                    
                    if chunk_pos in self.chunk_manager.chunks:
                        chunk   = self.chunk_manager.chunks[chunk_pos]
                        block_x = x - chunk_x * self.chunk_manager.chunk_size
                        block_z = z - chunk_z * self.chunk_manager.chunk_size
                        
                        # Check if block coordinates are valid
                        if (0 <= block_x < chunk.size_x and 0 <= y < chunk.size_y and 0 <= block_z < chunk.size_z):
                            block_type = chunk.blocks[block_x, y, block_z]
                            if block_type > 0:  blocks[(x, y, z)] = block_type
        

        return blocks
    
    def get_collision_box(self, position):
        # TODO: fix block offset issue
        # this seems to be working, should fix offset in the future
        # along with block raycasting offset!!   # offset=0.5
        min_x = position[0] - self.player_width  /2.5
        max_x = position[0] + self.player_width  *2
        min_y = position[1]
        max_y = position[1] + self.player_height
        min_z = position[2] - self.player_width  /2.5
        max_z = position[2] + self.player_width  *2
        
        return (min_x, max_x, min_y, max_y, min_z, max_z)
    
    def check_collision_with_block(self, collision_box, block_pos, block_type):
        if block_type == 0: return False # or block_type == 9: # TODO: water stuff
            
            
        min_x, max_x, min_y, max_y, min_z, max_z = collision_box
        bx, by, bz = block_pos
        
        # block bounds (cube units)
        b_min_x, b_max_x = bx, bx + 1
        b_min_y, b_max_y = by, by + 1
        b_min_z, b_max_z = bz, bz + 1
        
        # AABB collision test
        return (
            min_x < b_max_x - self.collision_buffer and 
            max_x > b_min_x + self.collision_buffer and
            min_y < b_max_y - self.collision_buffer and 
            max_y > b_min_y + self.collision_buffer and
            min_z < b_max_z - self.collision_buffer and 
            max_z > b_min_z + self.collision_buffer
        )
    
    def check_collision(self, position, blocks):
        if self.is_flying: return False 
        
        # collision box (cuboid)
        collision_box = self.get_collision_box(position)
        for block_pos, block_type in blocks.items():
            if self.check_collision_with_block(collision_box, block_pos, block_type):
                return True
        
        return False
    
    def check_ground(self, position, blocks):
        if self.is_flying: return False
        
        # bake 5x5 point grid below hitbox
        points = []
        for     x_offset in [-self.player_width/3, -self.player_width/6, 0, self.player_width/6, self.player_width/3]:
            for z_offset in [-self.player_width/3, -self.player_width/6, 0, self.player_width/6, self.player_width/3]:
                check_point_x = position[0] + x_offset
                check_point_z = position[2] + z_offset
                
                # Check at exact foot level and slightly below
                for y_offset in [0, -self.feet_offset ]:
                    check_point_y = position[1] + y_offset

                    block_x = int(check_point_x)
                    block_y = int(check_point_y)
                    block_z = int(check_point_z)
                    
                    points.append((block_x, block_y, block_z))
        
        # check for solids
        for point in points:
            if point in blocks and blocks[point] > 0 and blocks[point] != 9:  return True
                
        
        # Prevent 3 edge cases
        # new collision box below feet
        feet_collision_box = (
            position[0] - self.player_width/2,
            position[0] + self.player_width/2,
            position[1] - self.feet_offset ,
            position[1],
            position[2] - self.player_width/2,
            position[2] + self.player_width/2
        )
        
        for block_pos, block_type in blocks.items():
            if self.check_collision_with_block(feet_collision_box, block_pos, block_type): return True
                
        return False
    
    def get_penetration_depth(self, position, blocks):
        if self.is_flying:  return (0, 0, 0)
            
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_collision_box(position)
        penetration = [0, 0, 0]
        
        for block_pos, block_type in blocks.items():
            if block_type == 0: continue #or block_type == 9
                
            # block bounds
            bx, by, bz = block_pos
            
            b_min_x, b_max_x = bx, bx + 1
            b_min_y, b_max_y = by, by + 1
            b_min_z, b_max_z = bz, bz + 1
            
            # check overlapping
            if (min_x < b_max_x and max_x > b_min_x and
                min_y < b_max_y and max_y > b_min_y and
                min_z < b_max_z and max_z > b_min_z):
                
                # minimum penetration on each axis
                pen_x = min(max_x - b_min_x, b_max_x - min_x)
                pen_y = min(max_y - b_min_y, b_max_y - min_y)
                pen_z = min(max_z - b_min_z, b_max_z - min_z)
                
                if pen_x <= pen_y and pen_x <= pen_z:
                    if position[0] < bx + 0.5: 
                          penetration[0] = min(penetration[0], -pen_x) if penetration[0] < 0 else -pen_x
                    else: penetration[0] = max(penetration[0], pen_x)  if penetration[0] > 0 else  pen_x
                elif pen_y <= pen_x and pen_y <= pen_z:
                    if position[1] < by + 0.5:
                          penetration[1] = min(penetration[1], -pen_y) if penetration[1] < 0 else -pen_y
                    else: penetration[1] = max(penetration[1], pen_y)  if penetration[1] > 0 else  pen_y
                else:
                    if position[2] < bz + 0.5:
                          penetration[2] = min(penetration[2], -pen_z) if penetration[2] < 0 else -pen_z
                    else: penetration[2] = max(penetration[2], pen_z)  if penetration[2] > 0 else  pen_z
        
        return tuple(penetration)
    
    def resolve_collision(self, original_pos, new_pos, blocks):
        if self.is_flying: return new_pos

        # Final result position - start with original position
        result_pos = np.copy(original_pos)
        


        movement = new_pos - original_pos
        movement_length = np.linalg.norm(movement) # | mov vector |
        
        if movement_length < 0.001:
            # just check collision at target position
            if not self.check_collision(new_pos, blocks): return new_pos
            else:  return original_pos
        
        # step along vector
        steps = max(2, min(int(movement_length * 3), 10))  # adaptive number of steps
        step_vector = movement / steps
        
        # test intermediates
        current_pos = np.copy(original_pos)
        
        # horizontal test mov - prevent walking through walls
        horizontal_step = np.array([step_vector[0], 0, step_vector[2]], dtype=np.float32)
        horizontal_target = np.array([new_pos[0], original_pos[1], new_pos[2]], dtype=np.float32)
        
        for i in range(steps):
            test_pos = current_pos + horizontal_step
            
            # dont overshoot the target
            if i == steps - 1:
                test_pos = np.array(
                    [
                        horizontal_target[0], 
                        original_pos[1], 
                        horizontal_target[2]
                    ], dtype=np.float32
                )
            
            # check collision at test
            if self.check_collision(test_pos, blocks):
                
                # x
                test_x = np.array([test_pos[0], current_pos[1], current_pos[2]], dtype=np.float32)
                if not self.check_collision(test_x, blocks):  current_pos[0] = test_x[0]
                
                # z
                test_z = np.array([current_pos[0], current_pos[1], test_pos[2]], dtype=np.float32)
                if not self.check_collision(test_z, blocks):  current_pos[2] = test_z[2]
                    
                # stop on x,z fail : blocked
                if np.array_equal(current_pos, result_pos): break
                    
                result_pos = np.copy(current_pos)
            else:
                
                current_pos = np.copy(test_pos)
                result_pos = np.copy(current_pos)
        
        # vertical test
        vertical_step   = np.array([0, step_vector[1], 0], dtype=np.float32)
        vertical_target = np.array([result_pos[0], new_pos[1], result_pos[2]], dtype=np.float32)
        
        for i in range(steps):
            test_pos = current_pos + vertical_step
            
            if i == steps - 1: test_pos = vertical_target
            
            # check collision at test
            if self.check_collision(test_pos, blocks):
                # verticals
                if vertical_step[1] < 0:  #  down
                        self.on_ground = True
                        self.velocity[1] = 0  # stoop falling
                else:   self.velocity[1] = 0  # moving up, stop rising
                break
            else:
                current_pos = np.copy(test_pos)
                result_pos = np.copy(current_pos)
        
        # push out from inside block
        if self.check_collision(result_pos, blocks):
            # try to push upward
            test_up = np.array(
                [
                    result_pos[0], 
                    result_pos[1] + 0.1, 
                    result_pos[2]
                ], dtype=np.float32
            )
            if not self.check_collision(test_up, blocks):
                result_pos     = test_up
                self.on_ground = True
                self.velocity[1] = 0
            else:
                # upward blocked, test push for each axis
                for axis in range(3):
                    for direction in [-1, 1]:
                        push_pos  = np.copy(result_pos)
                        push_pos[axis] += 0.2 * direction
                        if not self.check_collision(push_pos, blocks):
                            result_pos = push_pos
                            if axis == 1 and direction == 1:  # if pushed up
                                self.on_ground = True
                                self.velocity[1] = 0
                            break
        
        return result_pos
    
    
    def update(self, delta_time, camera_front, camera_up, keys_pressed=[]):
        
        blocks = self.get_surrounding_blocks(self.position)
        
        
        self.on_ground = self.check_ground(self.position, blocks)
        self.can_jump  = self.on_ground or self.is_flying
        
        # timestep physics simulation to prevent tunneling
        remaining_time = delta_time
        num_steps      = 0
        
        while remaining_time > 0 and num_steps < self.max_physics_steps:
            step_time       = min(remaining_time, self.physics_step_size)
            remaining_time -= step_time
            num_steps      += 1
            

            original_pos = np.copy(self.position)
            
            movement = np.zeros(3, dtype=np.float32)
            speed = self.flight_speed if self.is_flying else self.movement_speed
            
            # cam relative movement
            horizontal_movement = np.zeros(3, dtype=np.float32)
            
            # ws movement
            if self.moving_forward or self.moving_backward:
                # get normalized horizontal vector of camera direction
                forward = np.array([camera_front[0], 0, camera_front[2]], dtype=np.float32)
                forward_length = np.linalg.norm(forward)
                
                if forward_length > 1e-6:  # Avoid division by zero with a small epsilon
                    forward    = forward / forward_length
                    dir_factor = 1.0 if self.moving_forward else -1.0
                    horizontal_movement += forward * dir_factor
            
            # ad movement
            if self.moving_left or self.moving_right:
                # get normalized right vector
                right = np.cross(camera_front, camera_up)
                right_length = np.linalg.norm(right)
                
                if right_length > 1e-6:
                    right = right / right_length
                    dir_factor = 1.0 if self.moving_right else -1.0
                    horizontal_movement += right * dir_factor
            
            # normalize combined xz vector
            horizontal_length = np.linalg.norm(horizontal_movement)
            if horizontal_length > 1e-6:
                horizontal_movement = horizontal_movement / horizontal_length
                
                movement[0] = horizontal_movement[0] * speed * step_time
                movement[2] = horizontal_movement[2] * speed * step_time
            
            # physics
            if not self.is_flying:
                # gravity
                self.acceleration[1] = self.gravity
                self.velocity += self.acceleration * step_time
                
                # term velocity
                if  self.velocity[1] < self.terminal_velocity:
                    self.velocity[1] = self.terminal_velocity
                    
                # jump
                current_time = time.time()
                if self.jumping and self.can_jump and current_time - self.last_jump_time > self.jump_cooldown:
                    self.velocity[1]    = self.jump_force
                    self.last_jump_time = current_time
                    self.on_ground = False
                
                # friction
                friction = self.friction if self.on_ground else self.air_friction
                self.velocity[0] *= friction
                self.velocity[2] *= friction
                
                # mov velocity
                movement += self.velocity * step_time
            else:
                # in flight mode
                if self.moving_up:   movement[1] += speed * step_time
                if self.moving_down: movement[1] -= speed * step_time
            


            # update position
            new_position = self.position + movement
            
            self.position = self.resolve_collision(original_pos, new_position, blocks)
            self.on_ground = self.check_ground(self.position, blocks)
            
            # if grounded, zero out vertical velocity to prevent jitter
            if self.on_ground and self.velocity[1] < 0: self.velocity[1] = 0
        

        return self.position




        