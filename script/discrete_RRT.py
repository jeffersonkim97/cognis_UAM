# Import Computation libraries
import numpy as np
import math
import random as rn
import copy

# Import Visualization libraries
import matplotlib.pylab as plt
import matplotlib.animation
from tempfile import NamedTemporaryFile
from IPython.display import HTML

def wrap2pi(x):
    """
    Wraps angle between 0 and 2pi
    """
    return x%(2*np.pi)

class Building:
    """
    This is a class deals with building setup and position
    biulding: dictionary containing building number, and their corners
    """
    def __init__(self, corner_vec):
        self.corner_vec = corner_vec

    def gen_building(self):
        """
        Generate dictionary containing following keys:
        nV: number of building corners
        cB: corner positions for buildings
        """
        building = {}
        building = {"nV": len(self.corner_vec), 
                    "cB": self.corner_vec}

        return building




class Camera:
    """
    This is a class deals with camera setup

    Variables
    cam_pos: Camera position
    cam_direc: Direction where the center line of camera FOV is facing at tf

    Camera Hardware constants (Later to be updated with real hardware)
    tilt_limit: Rotation angle limit for camera
    fov_ang: FOV angle (in radian)
    fov_rng: Camera detection range (in m)
    """
    def __init__(self, camx, camy, cam_direc):
        """
        camx: x position of camera
        camy: y position of camera
        cam_direc: direction of camera angle at tf
        """
        self.camx = camx
        self.camy = camy
        self.cam_direc = cam_direc

    def hardware_spec(self):
        """
        Hardware specification of cameras
        """
        self.tilt_limit = [0, np.pi]
        self.fov_ang = np.deg2rad(20) # [rad]
        self.fov_rng = 0.2            # [m]

    def camera_setup(self, n):
        """
        n: number of grid to construct n x n grid map, used for discreteMap
        """
        rot_vel = 2*(self.tilt_limit[1]-self.tilt_limit[0])/n # [rad/unit time]
        endtime = int(self.tilt_limit[1]*2/rot_vel)
        
        # Initialize empty storages
        ang_lim = [self.cam_direc-(self.tilt_limit[1]-self.tilt_limit[0])/2, self.cam_direc+(self.tilt_limit[1]-self.tilt_limit[0])/2]
        start_ang = ang_lim[0]
        tvec = np.linspace(0, endtime-1, endtime)
        camvec = self.fov_cam(start_ang, ang_lim, rot_vel, tvec)

        # Camera Position
        campos = np.array([self.camx, self.camy]).transpose()

        # Generate camera object dictionary
        cam_dict = self.gen_cam_obj(campos, rot_vel, start_ang, endtime, tvec, ang_lim, camvec)
        
        return cam_dict

    def fov_cam(self, start_ang, ang_lim, rot_vel, t):
        # Compute Camera angle (centerline of FOV) at each timestamp t
        ang_prev = start_ang
        fov_ang_vec = np.zeros(len(t)-1)
        dt = t[1]-t[0]
        
        # Iteration until it hits the angle limit
        iter_lim = 2*(ang_lim[1]-ang_lim[0])/rot_vel
        for i in range(len(t)-1):
            if i < iter_lim/2:
                fov_ang_curr = ang_prev + rot_vel*dt
            # Camera rotates back once reached its rotation limit
            else:
                fov_ang_curr = ang_prev - rot_vel*dt
            ang_prev = fov_ang_curr
            
            # Store the computed current angle into storage vector
            fov_ang_vec[i] = fov_ang_curr
        return fov_ang_vec

    def gen_cam_obj(self, campos, rot_vel, start_ang, endtime, tvec, ang_lim, camvec):
        cam_dict = {
            "pos": campos,
            "pos_time": camvec,
            "ang_init": start_ang,
            "rot_vel": rot_vel,
            "time_vec": tvec,
            "ang_lim": ang_lim,
            "tf": endtime,
            "spec": [self.tilt_limit, self.fov_ang, self.fov_rng]
        }
        return cam_dict




class DiscreteMap:
    """
    This is a discrete map generator, builds discrete map by combining static and dynamic map
    Static map is a map with stationary features, including building and non-moving obstacles
    Dynamic map is a map with mobile obstacles
    """
    def __init__(self, n, i_xi, i_xf):
        """
        n: number of grid to construct n x n grid map
        i_xi: Initial position index
        i_xf: Final position in index
        buildling: Dictionary which contains corners of buildings
        camera: Dictionary which contains position
        """
        self.n = n
        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)
        self.i_xi = i_xi
        self.i_xf = i_xf

    def position(self, i_x):
        """
        Convert grid index to position
        """
        return np.array([self.x[i_x[0]], self.y[i_x[1]]])


    def gen_grid(self, n):
        """
        Generate n x n grid map
        """
        X, Y = np.meshgrid(self.x, self.y)


    def graph_map(self, building_group, camera_group):
        """
        Display discrete map
        
        Input:
        building_group, camera_group: pre_built vector of building/camera obj
        """
        # Plot Map
        fig = plt.figure()
        
        # Plot buildings
        nB = building_group['nB']
        for iB in range(nB):
            curr_b = building_group[str(iB)]
            nV_iB = curr_b['nV'] # Number of corners
            nB_iB = curr_b['cB'] # Corner positions

            # Plot building polygon
            for ii in range(nV_iB/2):
                xB1 = ii
                yB1 = ii+nV_iB/2
                xB2 = xB1+1
                yB2 = yB1+1
                # If end of the array reached, loop back to close building polygon
                if xB2 >= nV_iB/2:
                    xB2 = xB2-nV_iB/2
                    yB2 = yB2-nV_iB/2
                plt.plot([nB_iB[xB1], nB_iB[xB2]], [nB_iB[yB1], nB_iB[yB2]], '-k', label='Building')

        # Plot camera
        nCam = camera_group['nCam']
        for iC in range(nCam):
            curr_cam = camera_group[str(iC)]
            cam_pos = curr_cam['pos']

            # Camera Position
            plt.plot(cam_pos[0], cam_pos[1], 'ok', label='Camera')
            
            # Camera FOV
            cam_spec = curr_cam['spec']
            fov_ang = cam_spec[1]
            fov_rng = cam_spec[2]
            theta = curr_cam['pos_time'][0]
            xend = cam_pos[0]+fov_rng*np.cos(theta)
            yend = cam_pos[1]+fov_rng*np.sin(theta)
            plt.plot([cam_pos[0], xend], [cam_pos[1], yend], '--k')

        # Box the grid map edges
        # Horizontal
        plt.plot([self.x[0], self.x[-1]], [self.y[0], self.y[0]], '-k', linewidth=3)
        plt.plot([self.x[0], self.x[-1]], [self.y[-1], self.y[-1]], '-k', linewidth=3)
        # Vertical
        plt.plot([self.x[0], self.x[0]], [self.y[0], self.y[-1]], '-k', linewidth=3)
        plt.plot([self.x[-1], self.x[-1]], [self.y[0], self.y[-1]], '-k', linewidth=3)

        # Plot Settings
        plt.grid()
        plt.axis('square')
        tol = 0.1
        plt.xlim(-tol, self.x[-1]+tol)
        plt.ylim(-tol, self.y[-1]+tol)




class DiscreteRRT:
    """
    This is a discrete RRT generator, using dynamic programming to find optimal path inbetween vertex
    Input: discrete Map, end time
    Output: Generated RRT
    """




#%%
if __name__ == "__main__":
    import numpy as np

    # Corner vectors
    corner = np.array([
        [0.1, 0.3, 0.25, 0.75],
        [0.7, 0.9, 0.25, 0.75]
    ])

    
    # Generate Building group
    bGroup = []
    for icor in range(len(corner)):
        icorner = corner[icor]
        print(icorner)

        iB = Building(icorner)
        bGroup.append(iB)

    print(bGroup)
# %%
