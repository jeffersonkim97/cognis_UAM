import numpy as np
from shapely import geometry 

# Camera object class
# This is a class for single camera setup, which returns
# 1) Position of camera mounted, 2) FOV area at given time t
class Camera:
    def __init__(self, cam_dict, x0, y0, z0):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.a = cam_dict['a']
        cam_spec = cam_dict['spec']
        self.tilt_lim = cam_spec['bound']
        self.fov_ang = cam_spec['fov'][0]
        self.Rc = cam_spec['fov'][1]
        self.cam_period = cam_spec['cam_time'][0]
        self.cam_dt = cam_spec['cam_time'][1]

    def cam_position(self):
        return (self.x0, self.y0, self.z0)
    
    def cam_radius(self):
        return self.Rc
    
    def cam_FOV(self):
        return self.fov_ang
    
    def get_ctr_theta_t(self, t_in):
        # Compute angle of FOV centerline, bounded between two tilt limits
        # This is a continuous function, and returns centerline angle at that given t_in
        up = self.tilt_lim[0]
        down = self.tilt_lim[1]
        A = np.abs(up - down)
        h = A/2
        B = 2*np.pi/self.cam_period

        return A/2*np.sin(B*t_in)#+h
    
    def get_fov(self, x0, y0, z0, t_in):
        # Compute FOV at given time t_in
        th = self.get_ctr_theta_t(t_in)
        d = self.Rc*np.cos(self.fov_ang/2)

        fov = np.vstack(([x0, y0, z0], [x0+d, y0, z0]))

        # Translate to origin
        trans = np.vstack([[x0, y0, z0],[x0, y0, z0]])
        fov -= trans

        # Rotate to match boundaries
        R3 = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th), np.cos(th), 0],
            [0, 0, 1]
        ])
        R2 = np.array([
            [np.cos(-self.a), 0, np.sin(-self.a)],
            [0, 1, 0],
            [-np.sin(-self.a), 0, np.cos(-self.a)],
        ])

        #print('rotation: ', np.rad2deg(th))
        
        # Rotate
        fov = fov@R3
        fov = fov@R2

        # Translate to position
        fov += trans

        return fov

    """def gen_fov_polygon(self, x0, y0, t_in):
        fov = self.get_fov(x0, y0, t_in)

        fov_list = []
        for i in range(3):
            fov_list.append(fov[i,:])

        return geometry.Polygon(geometry.LineString(fov_list)) """