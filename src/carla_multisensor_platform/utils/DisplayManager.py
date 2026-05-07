import pygame

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("CARLA Simulation")
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self.surfaces = {}

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]
    
    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]
    
    def get_display_offset(self, grid_position):
        display_size = self.get_display_size()
        return [int(grid_position[1] * display_size[0]), int(grid_position[0] * display_size[1])]
    
    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def add_surface(self, name, surface, grid_position):
        """Add a surface to be displayed at a specific grid position"""
        self.surfaces[name] = {'surface': surface, 'position': grid_position}

    def update_surface(self, name, new_surface):
        """Update an existing surface with new content"""
        if name in self.surfaces:
            self.surfaces[name]['surface'] = new_surface

    def get_sensor_list(self):
        return self.sensor_list
    
    def destroy_sensor(self):
        print("\n")
        for sensor in self.sensor_list:
            sensor.destroy_sensor()
        self.sensor_list = []
        self.surfaces.clear()
        
    def render_enable(self):
        return self.display is not None

    def render(self):
        if not self.render_enable():
            return

        for sensor in self.sensor_list:
            sensor.render()

        # Render surfaces
        for name, surface_data in self.surfaces.items():
            offset = self.get_display_offset(surface_data['position'])
            self.display.blit(surface_data['surface'], offset)

        pygame.display.flip()
