import numpy as np
from Camera import Camera

# Dynamic Map Generator
# 
class DynamicMap:
    def __init__(self, map_dict, cam_dict):
        self.period = map_dict['n']
        self.ncam = map_dict['ncam']
        self.cam_dict = cam_dict
        self.cam_x = cam_dict['x']
        self.cam_y = cam_dict['y']

    def gen_cam(self, t_in):
        # Generate n number of camera objects at input time t_in
        ncam = self.ncam

        Cam_on_field = {}
        for nc in range(ncam):
            # Variables
            x0=self.cam_x[nc]
            y0=self.cam_y[nc]

            # Generate Camera
            cam_nc = Camera(self.cam_dict, x0, y0)
            Cam_on_field[str(nc)] = {}
            Cam_on_field[str(nc)]['Camera'] = cam_nc
            Cam_on_field[str(nc)]['FOV'] = cam_nc.get_fov(x0, y0, t_in)
            Cam_on_field[str(nc)]['FOV_Poly'] = cam_nc.gen_fov_polygon(x0, y0, t_in)
        return Cam_on_field