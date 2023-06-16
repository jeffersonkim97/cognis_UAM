# Code Refactory for R2T2 code
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from shapely import geometry

# 3D Plot
from mpl_toolkits import mplot3d

# Custom Classes
from Dynamic import DynamicMap

# LieGroup Classes
import sys
sys.path.insert(1, '/home/kestrel1311/git/ros2_ws/src/cognis_UAM/script/LieGroup')
from SE3 import se3, SE3
from SO3 import so3, DCM, Euler, MRP, Quat
from SE2 import se2, SE2
from SO2 import so2, SO2


# R2T2 Code
class R2T2:
    
    def __init__(self, Xinit, Xfin, tvar, vehicle, map_in):
        # Xinit    = Initial Position
        # Xfin     = Final Position
        self.Xinit = Xinit
        self.Xfin  = Xfin
        
        # Time Variables
        self.trange  = tvar[0]
        self.delt    = tvar[1]
        self.endtime = tvar[2]

        # Vehicle Dictionary
        # 'v'         = Max velocity
        # 'radius'    = Radius
        # 'w'         = Max angular rate
        self.vmax     = vehicle['v']
        self.vradius  = vehicle['radius']
        self.vangular = vehicle['w']

        # Map Dictionary
        # 'n'              = Repeat period length
        # 'ncam'           = Number of sensors deployed
        # 'st'             = Static map
        # ['st']['size']   = 4x1 vector [xmin, xmax, ymin, ymax]
        # ['st']['n']      = Number of buildings
        # ['st'][str(i)]   = Building polygon of index i
        # 'dy'             = Periodic dynamic map
        # ['dy'][str(i)]   = Discrete dynamic map at index i
        # 'curr'           = Current map, combination of static and dynamic
        # ['curr'][str(i)] = Current map at index i

        self.map_st = map_in['st']
        self.map_dy = map_in['dy']

    def R2T2_2D(self):
        """
        Function for 2D RRT with Time Domain
        1. Generate sample time t
        2. Compute range R vehicle can reach
        3. Randomly sample a node Q
            80% to move to new node
            10% to remain stationary
            10% to move to final point
        4. If Q in cone connect. If not, resample
        5. Iterate until destination reached
        """

        # Empty storages for intermediate steps
        G = {}
        G['vertex'] = []
        G['neighbor'] = {}
        G['edge'] = {}
        G['route'] = []
        G['t'] = []

        # Initialize Parameters
        xi = self.Xinit
        xf = self.Xfin
        dt = self.delt
        trange = self.trange
        
        # Conversion to SE2
        Xi = SE2(xi)
        Xf = SE2(xf)
        
        # Initialize RRT Variables
        # Initial Node
        Qnear = SE2(xi)
        G['vertex'].append(Qnear)
        G['t'].append(0)
        Qnear_prev = G['vertex'][-1]
        # Initial Time
        tprev = 0
        # Map size
        map_size = self.map_st['size']
        # Dynamic Map
        dmap = self.map_dy
        # Storage for illegal path
        bad_path = []
        # Iteration counter
        counter = 0

        
        # Start R2T2 Loop, and iterate until destination reached
        # Controlling variables
        # Repeat R2T2 for another iteration?
        repeat_R2T2 = True
        # Turn on plot generation?
        plot = True

        # End conditon
        dist_best = self.get_range(Xi, Xf)
        dist_tol = 1e-1

        # Plot Setup
        if plot:
            fig = plt.figure(figsize=(7.5, 7.5))
            ax = fig.add_subplot(111,projection='3d')
            
            # Buildings
            for ib in range(self.map_st['n']):
                ibuilding = self.map_st[str(ib)]
                wall = geometry.LineString(ibuilding)
                building = geometry.Polygon(wall)
                bx, by = building.exterior.xy
                plt.plot(bx, by, 0, '-k', label='Building')

            # Initial Sensor Deployment
            cameras_at_t0 = dmap.gen_cam(G['t'][-1])
            for ic in range(cameras_at_t0['n']):
                camera_i_FOV = cameras_at_t0[str(ic)]['FOV']
                camera_i_FOV_Poly = camera_i_FOV[str(ic)]['FOV_Poly']
                # Plot Camera Location
                FOV_Poly_xi_t0, FOV_Poly_yi_t0 = camera_i_FOV_Poly.exterior.xy
                plt.plot(FOV_Poly_xi_t0, FOV_Poly_yi_t0, 0, '-g')

        # Start R2T2 Iteration
        while repeat_R2T2:
            # Step 0: Setup
            print('===========================')
            print('counter: ', counter)

            # Step 1: Generate sample time t
            ti = rn.uniform(0, trange)

            # Step 2: Compute movement range
            mv_R = self.vmax*ti

            # Step 3: Select movement
            # Currently we don't do weighted sampling
            switch = [0.8, 0.1, 0.1]
            Qnear = None
            Qnear_exist = False
            while not Qnear_exist:
                choose = rn.uniform(0,1)
                if choose >= 0 and choose <= switch[0]:
                    print('Random Node')
                    # Sample random point
                    Qnext = self.gen_node(mv_R, Qnear_prev.to_vec, tprev)
                    Qnear_exist = True
                elif choose > switch[0] and choose <= (switch[0]+switch[1]):
                    print('Stationary')
                    # Sample stationary point
                    Qnext = Qnear_prev
                    Qnear_exist = True
                else:
                    print('Final')
                    Qnext = Xf
                    Qnear_exist = True
            
            # Step 4
            # Find closest point (RRT* Implementation)
            Qnear, tnear = self.nearest(G, Qnext.to_vec)
            tnext = tnear + ti

            # Print out variables to check
            print('Qnear :', Qnear.to_vec)
            print('Qnext :', Qnext.to_vec)
            print('ti    : ', ti)
            st_tvec = np.arange(tnear, tnext, self.delt)

            # Step 5
            # Generate path and check collision
            Qroute = self.local_path_planner(Qnear, Qnext, self.vmax*ti)

            # Check collision
            if self.collision_check(self.vrad, Qnear, Qroute, st_tvec):
                bad_path.append(Qnext)
                print("Route Collision Detected")
            else:
                Qcurr, path_curr = self.current_pos(Qnear, Qnext, Qroute, st_tvec)
                print('Qcurr: ', Qcurr.to_vec)

                # Step 6
                # Check if destination reached
                a = Qnear.to_vec
                b = Qcurr.to_vec




        
    # Functiosn for Step 3
    def get_range(self, X0, X1):
        diff = X1.to_matrix - X0.to_matrix
        return np.sqrt((diff[0,2])**2 + (diff[1,2])**2)

    def gen_node(self, r, qnear, t_in):
        # Generate qrand in polar coordinate and convert to cartesian
        qrand_r = rn.uniform(0, r)
        qrand_th = rn.uniform(0, 2*np.pi)
        qrand_x = qrand_r*np.cos(qrand_th)
        qrand_y = qrand_r*np.sin(qrand_th)
        qrand = np.array([qrand_x + qnear[0], qrand_y + qnear[1], qrand_th])
        Qrand_SE2 = SE2(qrand)

        # Check if random sample is valid
        if self.static_bound(qrand) and not Qrand_SE2 is None and not self.dynamic_bound(qrand, t_in):
            return Qrand_SE2
        else:
            self.gen_node(r, qnear, t_in)

    def static_bound(self, xi):
        # Check if random sample's validity
        # True:  It is valid
        # False: It is not valid
        return self.in_map(xi) and not self.building_bound(xi)
        
    def in_map(self, xi):
        # Check if point is in map
        # True:  Point in map
        # False: Point outside of 
        return (xi[0] >= self.map_st['size'][0] and xi[0]<=self.map_st['size'][1] and xi[1] >= self.map_st['size'][2] and xi[1]<=self.map_st['size'][3])

    def building_bound(self, xi):
        # Check if point is inside of building
        # True:  Point inside building
        # False: Point outside of building
        smap = self.map_st
        check_vec = []
        for i in range(smap['n']):
            ibuildling = smap[str(i)]
            wall = geometry.LineString(ibuildling)
            building = geometry.Polygon(wall)

            check_vec.append(building.contains(geometry.Point(xi[0], xi[1])))
        return np.sum(check_vec)
    
    def dynamic_bound(self, xi, t_in):
        # Check collision with dynamic obstacles
        # True: Point inside one of camera FOV
        # False: Point outside of all camera FOV
        cameras = self.map_dy.gen_cam(t_in)
        ncam = cameras['n']
        check_vec = []
        for i in range(ncam):
            cam_i = cameras[str(i)]['FOV_Poly']
            check_vec.append(cam_i.contains(geometry.Point(xi[0], xi[1])))
        return bool(np.sum(check_vec))
    
    # Functions for Step 4
    def nearest(self, G, xi):
        range_vec = []
        for ii in range(len(G['vertex'])):
            VERT_avail = G['vertex'][ii]
            vert_avail = VERT_avail.to_vec
            temp = G['t'][ii]
            range_vec.append(np.sqrt((xi[0]-vert_avail[0])**2 + (xi[1]-vert_avail[1])**2))
        Xnear = G['vertex'][range_vec.index(min(range_vec))]
        t_at_Xnear = G['t'][range_vec.index(min(range_vec))]

        return Xnear, t_at_Xnear
    
    # Functions for Step 5
    def local_path_planner(self, X0, X1, dist):
        # Plot route from start X0 to end X1
        u, R, d = self.find_u_R_d(X0, X1)
        if np.abs(u) > dist:
            u = dist*np.sign(u)
        
        # Compute turning angle
        if np.isclose(np.abs(R), 0):
            omega = 0
        else:
            omega = u/R

        v = se2(np.array([u, 0, omega]))
        V = v.exp

    def find_u_R_d(self, X0, X1):
        # Compute arc length, radius, distance
        M = X0.inv @ X1.to_matrix
        dx = M.to_vec[0]
        dy = M.to_vec[1]
        dth = M.to_vec[2]
        
        d = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy, dx)
        
        if np.abs(alpha) > 1e-3:
            R = d/(2*np.sin(alpha))
            u = 2*R*alpha
        else:
            R = np.infty
            u = d

        return u, R, d

    def collision_check(self, vehicle_radius, Q0, Q1, qtvec):
        V = Q0.inv @ Q1.to_matrix
        v = V.log
        steps = len(qtvec)
        test_vector = np.linspace(0, 1, steps)
        for tt in test_vector:
            Q = Q0.to_matrix @ SE2(v*tt)
            xt = Q.to_vec[0]
            yt = Q.to_vec[1]
            thetat = Q.to_vec[2]

            if not self.in_map(Q.to_vec):
                if bool(self.building_bound(Q.to_vec)):
                    return True
            
            t_in_to_test, = np.where(test_vector == tt)
            t_to_test = qtvec[t_in_to_test[0]]
            if bool(self.dynamic_bound((xt, yt), t_to_test)):
                return True
            
        return False





#%%
if __name__=="__main__":
    import numpy as np
    from Dynamic import DynamicMap

    # Test case
    # Initial, Final Positions
    x0 = np.array([0, 0, 0])
    x1 = np.array([10, 0, 0])
    t = [50, .1, 100]

    # Vehicle Spec
    vehicle = {}
    vehicle['v'] = 1
    vehicle['radius'] = 0.5
    vehicle['w'] = np.deg2rad(30)

    # Map
    map_in = {}
    # Static Map
    map_in['st'] = {}
    map_in['st']['size'] = np.array([-5, 15, -15, 15])
    # Single buliding example
    map_in['st']['n'] = 1
    map_in['st']['0'] = np.array([
        (2.5,5), (7.5,5), (7.5,-5), (2.5,-5)
    ])

    # Dynamic Map
    # This is a continuous function generates camera FOV coverages
    # Input is map_in, and time input t_in
    map_in['n'] = t[0]
    map_in['ncam'] = 1

    # Single camera example, surveying final location xfin
    # Camera Position
    cam_x = np.array([7.5])
    cam_y = np.array([0])
    cam_dict = {}
    cam_dict['n'] = len(cam_x)
    cam_dict['x'] = cam_x
    cam_dict['y'] = cam_y

    # Camera Spec
    tilt_limit = np.array([np.pi, 0]) #[upper, lower]
    fov_ang = np.deg2rad(20)
    fov_rng = 7.5 #[m]
    cam_period = t[0]
    cam_increment = t[1]
    cam_dict['spec'] = {}
    cam_dict['spec']['bound'] = tilt_limit
    cam_dict['spec']['fov'] = [fov_ang, fov_rng]
    cam_dict['spec']['cam_time'] = [cam_period, cam_increment]
    
    # Test dynamic map
    dmap = DynamicMap(map_in, cam_dict)
    map_in['dy'] = dmap

    test = R2T2(x0, x1, t, vehicle, map_in)
    RRT = test.R2T2_2D()