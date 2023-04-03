# Import Computation libraries
import numpy as np
import math
import random as rn
import copy

# Import Visualization libraries
import matplotlib.pylab as plt
import matplotlib.animation
import matplotlib.path as mplPath
from tempfile import NamedTemporaryFile
from IPython.display import HTML

# Global Functions
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
        building = {'nV': len(self.corner_vec), 
                    'cB': self.corner_vec}

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
        self.hardware_spec()

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
            'pos': campos,
            'pos_time': camvec,
            'ang_init': start_ang,
            'rot_vel': rot_vel,
            'time_vec': tvec,
            'ang_lim': ang_lim,
            'tf': endtime,
            'spec': [self.tilt_limit, self.fov_ang, self.fov_rng]
        }
        return cam_dict

class DiscreteMap:
    """
    This is a discrete map generator, builds discrete map by combining static and dynamic map
    Static map is a map with stationary features, including building and non-moving obstacles
    Dynamic map is a map with mobile obstacles
    """
    def __init__(self, n, i_xi, i_xf, bG, cG):
        """
        n: number of grid to construct n x n grid map
        i_xi: Initial position index
        i_xf: Final position in index
        bG: Dictionary which contains corners of buildings
        cG: Dictionary which contains position
        """
        self.n = n
        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)
        self.i_xi = i_xi
        self.i_xf = i_xf
        self.bG = bG
        self.cG = cG

    def position(self, i_x):
        """
        Convert grid index to position
        """
        return np.array([self.x[i_x[0]], self.y[i_x[1]]])

    def gen_grid(self):
        """
        Generate n x n grid map
        """
        X, Y = np.meshgrid(self.x, self.y)
        return X, Y
    
    def bound(self, i_x):
        """
        Bound within map
        """
        return (i_x[0] >= 0 and i_x[0]<self.n and i_x[1] >= 0 and i_x[1]<self.n)
    
    def bound_cam(self, i_x, campos, fov_ang_curr, fov_ang, fov_rng):
        """
        Bound outside of camera FOV
        """
        # Convert i_x to position
        i_xPos = self.position(i_x)
        # distance to node
        dist = np.sqrt((campos[0]-i_xPos[0])**2 + (campos[1]-i_xPos[1])**2)
        # Angle to node
        ang = wrap2pi(np.arctan2(i_xPos[1]-campos[1], i_xPos[0]-campos[0]))
        
        return wrap2pi(fov_ang_curr+fov_ang/2)-ang>=0 and wrap2pi(fov_ang_curr-fov_ang/2)-ang<=0 and fov_rng - dist >= 0

    def bound_building(self, i_x, building_group):
        nB = len(building_group)
        currpos = self.position(i_x)
        bcheck = np.zeros((nB,1))
        """
        for b in range(nB):
            bnow = building_group[b]
            bcheck[b] = (currpos[0] >= bnow[0] and currpos[0] <= bnow[1] and currpos[1] >= bnow[2] and currpos[1] <= bnow[3])
        """
        for b in range(nB):
            bnow = mplPath.Path(building_group[b]['cB'])
            bcheck[b] = bnow.contains_point(currpos)
        sumcheck = 0
        for b in range(nB):
            sumcheck += bcheck[b]
        return sumcheck

    def graph_map(self, building_group, camera_group):
        """
        Display map. This is for sanity check
        
        Input:
        building_group, camera_group: pre_built vector of building/camera obj
        """
        # Plot Map
        fig = plt.figure()
        
        # Plot buildings
        nB = len(building_group)
        for iB in range(nB):
            curr_b = building_group[iB]
            nV_iB = curr_b['nV'] # Number of corners
            nB_iB = curr_b['cB'] # Corner positions

            # Plot building polygon
            for ii in range(int(nV_iB+1)):
                vert1 = int(ii)
                vert2 = int(ii+1)

                # If end of the array reached, loop back to close building polygon
                if vert1 >= int(nV_iB):
                    vert1 = int(ii-nV_iB)
                if vert2 >= int(nV_iB):
                    vert2 = int(ii+1-nV_iB)

                plt.plot([nB_iB[vert1][0], nB_iB[vert2][0]], [nB_iB[vert1][1], nB_iB[vert2][1]], '-k', label='Building')

        # Plot camera
        nCam = len(camera_group)
        for iC in range(nCam):
            curr_cam = camera_group[iC]
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
        plt.show()

    def graph_discreteMap(self, key, current_map):
        """
        Display discrete map. This is for sanity check
        """
        X, Y = self.gen_grid()

        fig = plt.figure()
        for i in range(len(X)):
            for j in range(len(Y)):
                if current_map[str(key)][i,j]:
                    plt.plot(X[0,i], Y[j,0], '.b')
        plt.show()

    def static_movement_map(self, building_group):
        """
        Generate static map
        """
        # Static Movement Map
        X, Y = self.gen_grid()

        nH = len(X)
        nV = len(X[0])
        
        # For (x, y) coordinate, check if this is within the field bound while outside of building bound
        static_map = np.zeros((nH, nV))
        for i in range(len(X)):
            for j in range(len(Y)):
                # Check if the (i,j) is within field bound
                within_bound = self.bound((i, j))
                # check if the (i,j) is outside building bound
                within_building = self.bound_building((i,j), building_group)
                
                # If we are within_bound but not in building >> True
                # else >> False
                static_map[i,j] = within_bound and not within_building
        return static_map
    
    def dynamic_movement_map(self, camera_group):
        """
        Generate dynamic map
        """
        # Dynamic Movement Map at time t from tvec
        # Obtain camera FOV position at time t
        X, Y = self.gen_grid()
        nH = len(X)
        nV = len(X[0])        
        
        # Extract necessary info
        n_cam = len(camera_group)

        dynamic_map = {}
        for tk in range(self.n-1):
            # At tk, we know where ith camera's angle theta_ik is facing
            currkey = str(tk)
            
            # Check if each grid cells are included in camera FOV
            # If outside camera FOV >> True
            # If inside camera FOV >> False
            map_k = np.zeros((nH, nV))
            for i in range(len(X)):
                for j in range(len(Y)):
                    camcheck = np.zeros(n_cam)
                    for k in range(n_cam):
                        # Extract Camera Info
                        curr_cam = camera_group[k]
                        campos = curr_cam['pos']
                        camvec = curr_cam['pos_time']
                        fov_ang = curr_cam['spec'][1]
                        fov_rng = curr_cam['spec'][2]

                        # Check if gird coordinate (i,j) is in kth camera's FOV
                        camcheck[k] = self.bound_cam((i,j), campos, camvec[tk], fov_ang, fov_rng)
                    map_k[i,j] = not camcheck.any()
                    
            dynamic_map[currkey] = map_k
        
        return dynamic_map
    
    def current_movement_map(self, static_map, dynamic_map):
        """
        Combine static and dynamic map together, to generate map at current time key stored in dictionary
        """
        # Take static map and add dynamic map at time tk
        # to generate current movement map
        X, Y = self.gen_grid()
        nH = len(X)
        nV = len(X[0])

        current_map = {}
        # At each tk, generate current map and store as dictionary
        for tk in range(self.n-1):
            currkey = str(tk)
            currmap_k = np.zeros((nH, nV))
            for i in range(len(X)):
                for j in range(len(Y)):
                    currmap_k[i,j] = static_map[i,j] and dynamic_map[currkey][i,j]
            current_map[currkey] = currmap_k
        return current_map

    

class DiscreteRRT:
    """
    This is a discrete RRT generator, using dynamic programming to find optimal path inbetween vertex
    Input: discrete Map, end time
    Output: Generated RRT
    """


#%%
if __name__ == "__main__":
    import numpy as np
    # Discrete Map Setup
    n = 51
    i_x0 = (int(0.05*n), int(0.05*n))
    i_xf = (int(0.95*n), int(0.95*n))

    # Corner vectors for buildings
    corner = np.array([
        [(0.1, 0.25), (0.3, 0.25), (0.3,0.75), (0.1,0.75)],
        [(0.7, 0.25), (0.9, 0.25), (0.9,0.75), (0.7,0.75)]
    ])

    # Generate Building group
    bGroup = []
    for icor in range(len(corner)):
        icorner = corner[icor]
        iB = Building(icorner).gen_building()
        bGroup.append(iB)

    # Camera Setup
    camx_vec = np.array([.3, .7, .3, .7])
    camy_vec = np.array([.25, .25, .75, .75])
    cam_direc_vec = np.array([np.pi*7/4, np.pi*5/4, np.pi*1/4, np.pi*3/4])

    # Generate Camera group
    nCam = len(camx_vec)
    cGroup = []
    for icam in range(nCam):
        iC = Camera(camx_vec[icam], camy_vec[icam], cam_direc_vec[icam]).camera_setup(n)
        cGroup.append(iC)
    
    # Generate Discrete Map
    dcMap = DiscreteMap(n, i_x0, i_xf, bGroup, cGroup)
    #dcMap.graph_map(bGroup, cGroup)
    sMap = dcMap.static_movement_map(bGroup)
    dyMap = dcMap.dynamic_movement_map(cGroup)
    currMap = dcMap.current_movement_map(sMap, dyMap)
    #dcMap.graph_discreteMap(0, currMap)

    # 
#%%