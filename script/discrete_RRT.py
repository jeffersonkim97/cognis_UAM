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

def rangeTo(q0, q1):
    """
    Measure range from q0 to q1
    """
    return np.sqrt((q1[0]-q0[0])**2 + (q1[1]-q0[1])**2)

# SE2
def SE2_log(M):
    """
    Matrix logarithm for SE2 Lie group
    """
    theta = np.arctan2(M[1, 0], M[0, 0])
    if np.abs(theta) < 1e-6:
        A = 1 - theta**2/6 + theta**4/120
        B = theta/2 - theta**3/24 + theta**5/720
    else:
        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta
    V_inv = np.array([[A, B], [-B, A]])/(A**2 + B**2)
    t = M[:2, 2]
    u = V_inv.dot(t)
    return np.array([theta, u[0], u[1]])

def SE2_from_param(v):
    """
    Create SE2 from paramters, [theta, x, y]
    """
    theta, x, y = v
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def SE2_to_param(M):
    """
    From matrix to [theta, x, y]
    """
    theta = np.arctan2(M[1, 0], M[0, 0])
    x = M[0, 2]
    y = M[1, 2]
    return np.array([theta, x, y])

def SE2_inv(M):
    """
    SE2 inverse
    """
    R = M[:2, :2]
    t = np.array([M[:2, 2]]).T
    return np.block([
        [R.T, -R.T.dot(t)],
        [0, 0, 1]
    ])

def SE2_exp(v):
    """
    SE2 matrix exponential
    """
    theta, x, y = v
    if np.abs(theta) < 1e-6:
        A = 1 - theta**2/6 + theta**4/120
        B = theta/2 - theta**3/24 + theta**5/720
    else:
        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta
    V = np.array([[A, -B], [B, A]])
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    u = np.array([[x, y]]).T
    return np.block([
        [R, V.dot(u)],
        [0, 0,  1]])


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
                    plt.plot(X[0,i], Y[j,0], '.k')
        plt.grid()
        plt.axis('square')
        tol = 0.1
        plt.xlim(-tol, self.x[-1]+tol)
        plt.ylim(-tol, self.y[-1]+tol)
        plt.show()

    def static_movement_map(self, building_group, camera_group):
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
        # Generate safety ring around camera position
        safety = 0.025
        for c in range(len(camera_group)):
            for i in range(len(X)):
                for j in range(len(Y)):
                    curr_cam = camera_group[c]
                    campos = curr_cam['pos']
                    if rangeTo(self.position((i,j)), campos) <= safety:
                        static_map[i,j] = False
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

    Lie Group coordinate X = SE2_from_param(theta,i_x[0],i_x[1])
    """
    def __init__(self, currMap, endtime, buildings, cameras):
        self.currMap = currMap
        self.endtime = endtime
        self.bGroup = buildings
        self.cGroup = cameras

    def bound_map(self, iX, submap_vertex):
        """
        Check if iX is bound within submap
        """
        return (iX[0,2] >= submap_vertex[0]
                and iX[0,2] < submap_vertex[1]+1
                and iX[1,2] >= submap_vertex[2]
                and iX[1,2] < submap_vertex[3]+1)

    def costfunc(self, X0, X1):
        """
        Cost function (Euclidean distance to goal) computed in SE2
        Input: X0, X1 are coordinate in SE2 Lie Group
        Output: Cost function
        """
        etheta, ex, ey = SE2_to_param(SE2_inv(X0).dot(X1))
        return np.sqrt(ex**2 + ey**2)
    
    def check_map(self, iX, current_map_k):
        """
        check if iX position in current map is valid
        """
        return bool(current_map_k[iX[0,2], iX[1,2]])
    
    def reverse_pos(self, pos):
        """
        Obtain index position from real-space position
        """
        xp = pos[0]
        yp = pos[1]

        xcost = 1
        ycost = 1
        for i in range(len(x)):
            if abs(x[i]-xp) < xcost:
                xcost = abs(x[i]-xp)
                ix = i
        for j in range(len(y)):
            if abs(y[i] - yp) < ycost:
                ycost = abs(y[j]-yp)
                iy = i

        return (ix,iy)
    
    def cost_func_propagate(self, X0, X1, mapsize, submap, submap_vertex, endK):
        """
        Propagate cost functoin between Lie Group coordinate X0 and X1
        Generated map size is (i,j)
        """
        # Initialize Final Point
        iX_prev = X0
        iX = iX_prev

        # Build Empty Propagation
        V = np.zeros((mapsize[0], mapsize[1]))

        # Map translation vector
        trans = [submap_vertex[0], submap_vertex[2]]

        # History Storage
        live = [iX[0,2], iX[1,2]]
        live_hist = {}
        live_hist['0'] = live

        # Actions Possible
        # (theta, x, y)
        movement = [
            (0, 1, 0), (np.pi, -1, 0), # Horizontal
            (np.pi/2, 0, 1), (3*np.pi/2, 0, -1), # Vertical
            (0, 0, 0), # Stationary
            (np.pi/4, 1, 1), (3/4*np.pi, -1, 1),
            (5/4*np.pi, -1, -1), (7/4*np.pi, 1, -1) # Diagonal
        ]

        # Compute moves from current node
        V_data = []
        time_fin = None
        counter = 0
        kek = endK[0]
        while len(live) > 0:
            new = set()
            # Bound iteration between camvec
            end_in_prev = kek
            if end_in_prev > len(camvec):
                end_in_prev = end_in_prev - len(camvec)
            elif end_in_prev < 0:
                end_in_prev = end_in_prev + len(camvec)
            currkey = str(end_in_prev)
        
            # Select current map
            cmap = submap[currkey]

            # Populate Node for cost function
            for p_SE2 in live:
                for a in movement:
                    a_SE2 = SE2_from_param(a)
                    pa_SE2 = p_SE2 - a_SE2

                    # If our next move (pa) is valid in current map:
                    if self.bound_map(pa_SE2, submap_vertex) and self.check_map(pa_SE2, currMap[currkey]):
                        # If pa is out of all camera FOV
                        if pa_SE2 == X1:
                            time_fin = counter
                        V_new = self.costfunc(p_SE2, pa_SE2)+V[p_SE2[0,2]-trans[0], p_SE2[1,2]-trans[1]]
                        V_old = V[pa_SE2[0,2]-trans[0], pa_SE2[1,2]-trans[1]]
                        if V_old == 0 or V_new < V_old:
                            V[pa_SE2[0,2]-trans[0], pa_SE2[1,2]-trans[1]] = V_new
                            new.add(pa_SE2)
            live = new
            key = str(counter)
            live_hist[key] = live
            V_data.append(copy.copy(V))
            kek = end_in_prev - 1
            counter += 1

        for i in range(len(V_data)):
            V[X0[0,2]-trans[0], X0[1,2]-trans[1]] = 0
            V_data[i][X0[0,2]-trans[0], X0[1,2]-trans[1]] = 0
        
        return V, V_data, live_hist
    
    def plot_course(self, X0, X1, V, V_data, submap_vertex, submap):
        # Actions Possible
        # (theta, x, y)
        movement = [
            (0, 1, 0), (np.pi, -1, 0), # Horizontal
            (np.pi/2, 0, 1), (3*np.pi/2, 0, -1), # Vertical
            (0, 0, 0), # Stationary
            (np.pi/4, 1, 1), (3/4*np.pi, -1, 1),
            (5/4*np.pi, -1, -1), (7/4*np.pi, 1, -1) # Diagonal
        ]

        # Path Validity Trigger
        path_good = False

        # Map translation vector
        trans = [submap_vertex[0], submap_vertex[2]]

        # Sanity Check
        SanityCheck = type(None)

        # Empty Storage Dictionary
        p_hist_opt_k = {}
        pos_hist_opt_k = {}

        # Compute Optimal Path for time vector start
        # Then we match final FOV to find out which time tk start grans camera final state at tf
        for tk in range(len(tvec.T)):
            p_SE2 = X1
            p_hist_opt = [p_SE2]
            count = 0

            # Unitl we reach X0
            while p_SE2 != X0:
                # Find current time to check where the camera angle is:
                curr_time = (tk+count)%len(camvec.T)
                currkey = str(curr_time)
                next_time = int(curr_time+1)
                if next_time >= len(camvec.T):
                    next_time = int(next_time - len(camvec))
                nextkey = str(next_time)

                # Populate Node for cost function
                V_opt = None
                Vcheck = []
                pa_list = []

                for a in movement:
                    # Movement action 'a' taken from starting point p
                    a_SE2 = SE2_from_param(a)
                    pa_SE2 = p_SE2 + a_SE2

                    # Check Camera State
                    # If our next move (pa) in current map in tk is valid,
                    pa_converted = SE2_from_param([0, pa_SE2[0,2]-trans[0], pa_SE2[1,2]-trans[1]])

                    if self.bound_map(pa_SE2, submap_vertex) and self.check_map(pa_converted, submap[nextkey]):
                        # If pa is out of all camera bound
                        V_new = V[pa_converted[0,2], pa_converted[1,2]]
                        pa_list.append(pa_SE2)
                        Vcheck.append(V_new)

                if len(Vcheck) == 0: break

                min_val = Vcheck.index(min(Vcheck))
                V_opt = Vcheck[min_val]
                pa_opt = pa_list[min_val]

                for a in movement:
                    a_SE2 = SE2_from_param(a)
                    if self.bound_map(pa_SE2, submap_vertex) and self.check_map(pa_converted, submap[nextkey]):
                        if pa_SE2 == X0: break
                
                # Exit loop if this is not going anywhere
                if count > 100: break
                p_SE2 = pa_opt

                # Check if p is empty
                # If p is empty: There is no solution exist s.t. x0 and xf can be connected
                if isinstance(p_SE2, SanityCheck):
                    break
                else:
                    p_hist_opt.append(p_SE2)
                    count += 1

            # Obtain position history of tk start
            pos_hist_opt = np.array([self.position(p) for p in p_hist_opt])

            # Save histories to dictionary with key tk
            p_hist_opt_k[str(tk)] = p_hist_opt
            pos_hist_opt_k[str(tk)] = pos_hist_opt
            path_good = True

        # If p_hist_opt is not empty
        # aka if we do have path from x0 to x1
        if path_good == True:
            # Find postion history with matching camera state at tf
            tf_vec = []
            for k in range(len(pos_hist_opt_k)):
                # k is a start time
                # If we add length of time taken to reach xf and take modular value, we get tf
                tf_k = (k + len(pos_hist_opt_k[str(k)]))%len(camvec.T)

                # check if obtained tf_k is same as our set tf
                if tf_k == end_in[0]:
                    tf_vec.append(tf_k)
            return p_hist_opt, pos_hist_opt_k, tf_vec
        else:
            p_hist_opt = None
            pos_hist_opt_k = None
            tf_vec = None
            return p_hist_opt, pos_hist_opt_k, tf_vec

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
    sMap = dcMap.static_movement_map(bGroup, cGroup)
    dyMap = dcMap.dynamic_movement_map(cGroup)
    currMap = dcMap.current_movement_map(sMap, dyMap)
    dcMap.graph_discreteMap(0, currMap)

    # 
#%%