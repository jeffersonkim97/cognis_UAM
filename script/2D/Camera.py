import numpy as np
from shapely import geometry 

# Camera object class
# This is a class for single camera setup, which returns
# 1) Position of camera mounted, 2) FOV area at given time t
class Camera:
    def __init__(self, i, cam_dict, x0, y0):
        self.x0 = x0
        self.y0 = y0
        self.i = i
        cam_spec = cam_dict['spec']
        self.tilt_lim = cam_spec['bound'][i]
        self.fov_ang = cam_spec['fov'][0]
        self.Rc = cam_spec['fov'][1]
        self.init_angle = cam_dict['spec']['init_angle'][i]
        self.cam_period = cam_spec['cam_time'][0]
        self.cam_dt = cam_spec['cam_time'][1]
        self.cam_fovspeed = cam_spec['panspeed'][i]

    def cam_position(self):
        return (self.x0, self.y0)
    
    def p1(self, x0, y0, th):
        x1 = x0 + self.Rc*np.cos(th)
        y1 = y0 + self.Rc*np.sin(th)
        return [x1, y1]
    
    def p2(self, x0, y0, th):
        x2 = x0 + self.Rc*np.cos(th)
        y2 = y0 - self.Rc*np.sin(th)
        return [x2, y2]
    
    def get_ctr_theta_t(self, t_in):
        # Compute angle of FOV centerline, bounded between two tilt limits
        # This is a continuous function, and returns centerline angle at that given t_in
        up = self.tilt_lim[0]
        down = self.tilt_lim[1]
        A = np.abs(up - down)
        h = A/2
        # B = 2*np.pi/self.cam_period
        B = 2*np.pi*self.cam_fovspeed

        return A/2*np.sin(B*t_in) + self.init_angle#+h
    
    def get_fov(self, x0, y0, t_in):
        # Compute FOV at given time t_in
        th = self.get_ctr_theta_t(t_in)
        p1 = self.p1(x0,y0,self.fov_ang/2)
        p2 = self.p2(x0,y0,self.fov_ang/2)

        fov = np.vstack(([x0, y0], p1, p2))

        # Translate to origin
        trans = np.vstack([[x0, y0],[x0, y0],[x0, y0]])
        fov -= trans

        # Rotate to match boundaries
        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])

        #print('rotation: ', np.rad2deg(th))
        
        # Rotate
        fov = fov@R

        # Translate to position
        fov += trans

        return fov

    def gen_fov_polygon(self, x0, y0, t_in):
        fov = self.get_fov(x0, y0, t_in)

        fov_list = []
        for i in range(3):
            fov_list.append(fov[i,:])

        return geometry.Polygon(geometry.LineString(fov_list))