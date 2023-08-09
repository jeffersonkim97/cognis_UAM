import numpy as np

class Camera:
    def __init__(self, cam_dict):
        # Mounted camera position and PT angle
        cam_pos_init = cam_dict['pos_init']
        cam_ang_init = cam_dict['ang_init']
        self.x0 = cam_pos_init[0]
        self.y0 = cam_pos_init[1]
        self.z0 = cam_pos_init[2]
        self.pan = cam_ang_init[0]
        self.tilt = cam_ang_init[1]

        cam_spec = cam_dict['spec']
        self.tilt_lim = cam_spec['ang_limit'][0]
        self.pan_lim = cam_spec['ang_limit'][1]
        self.FOV = cam_spec['FOV']
        self.R = cam_spec['range']
        self.cam_period = cam_spec['cam_time'][0]
        self.cam_dt = cam_spec['cam_time'][1]

    def get_state(self):
        return np.array([self.x0, self.y0, self.z0, self.pan, self.tilt])
    
    def get_direction_vec(self, pan, tilt):
        thetp = pan
        thett = tilt

        ctp = np.cos(thetp)
        stp = np.sin(thetp)
        ctt = np.cos(thett)
        stt = np.sin(thett)

        Ry = np.array([[ctt, 0, stt], [0, 1, 0], [-stt, 0, ctt]])
        Rz = np.array([[ctp, -stp, 0], [stp, ctp, 0], [0, 0, 1]])

        direc_vec = Ry@Rz@np.vstack((1,0,0))

        return direc_vec
    
    def get_h(self):
        return self.R
    
    def get_radius(self):
        return self.R*np.sin(self.FOV/2)
    
    def get_base(self, pan, tilt):
        radius = self.get_radius()
        height = self.get_h()

        angle_vector = np.linspace(0, 2*np.pi, 100)
        circle_ptx = 0*angle_vector
        circle_pty = radius*np.cos(angle_vector)
        circle_ptz = radius*np.sin(angle_vector)
        base = np.vstack((circle_ptx, circle_pty, circle_ptz))

        thetp = pan
        thett = tilt

        ctp = np.cos(thetp)
        stp = np.sin(thetp)
        ctt = np.cos(thett)
        stt = np.sin(thett)

        Ry = np.array([[ctt, 0, stt], [0, 1, 0], [-stt, 0, ctt]])
        Rz = np.array([[ctp, -stp, 0], [stp, ctp, 0], [0, 0, 1]])
        
        rotated_base = Ry@Rz@base
        translate = height*self.get_direction_vec(pan, tilt)+np.vstack((self.x0, self.y0, self.z0))
        return rotated_base + translate
    
    def get_pan_vec(self):
        pan_start = self.pan
        pan_end = (self.pan + self.pan_lim)%(2*np.pi)
        time_interval = self.cam_period/self.cam_dt

        return np.linspace(pan_start, pan_end, int(time_interval))
    
    def get_pan_instance(self, time_in):
        pan_start = self.pan
        pan_end = (self.pan + self.pan_lim)%(2*np.pi)
        pan_instance = (pan_end-pan_start)/self.cam_period*(time_in%self.cam_period)
        return pan_instance