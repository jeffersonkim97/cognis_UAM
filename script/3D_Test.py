# Code Refactory for R2T2 code
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from shapely import geometry

# 3D Plot
from mpl_toolkits import mplot3d

# Custom Classes
from Dynamic import DynamicMap


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
        self.yaw = vehicle['yaw']
        self.pitch = vehicle['pitch']

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
        
        
        # Initialize RRT Variables
        # Initial Node
        Qnear = xi
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
        dist_best = self.get_range(xi, xf)
        dist_tol = 1e-1

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
            switch = [0.8, 0.2]
            #switch = [1, 0, 0]
            Qnear = None
            Qnear_exist = False
            while not Qnear_exist:
                choose = rn.uniform(0,1)
                if choose >= 0 and choose <= switch[0]:
                    print('Random Node')
                    # Sample random point
                    Qnext = self.gen_node(mv_R, Qnear_prev, tprev)
                    Qnear_exist = True
                #elif choose > switch[0] and choose <= (switch[0]+switch[1]):
                #    print('Stationary')
                    # Sample stationary point
                #    Qnext = Qnear_prev
                #    Qnear_exist = True
                else:
                    print('Final')
                    Qnext = xf
                    Qnear_exist = True
            print('Qnext :', Qnext)

            # Step 4
            # Find closest point (RRT* Implementation)
            Qnear, tnear = self.nearest(G, Qnext)
            tnext = tnear + ti

            # Print out variables to check
            print('Qnear :', Qnear)
            print('ti    : ', ti)
            st_tvec = np.arange(tnear, tnext, self.delt)

            # Step 5
            # Generate path and check collision
            # Check collision
            if self.collision_check(Qnear, Qnext, st_tvec, len(st_tvec)):
                bad_path.append(Qnext)
                print("Route Collision Detected")
            else:
                Qcurr, path_curr = self.current_pos(Qnear, Qnext, Qroute, st_tvec)
                print('Qcurr: ', Qcurr)

            # Step 6
            # Check if destination reached
                a = Qnear
                b = Qcurr
                edge = [a[0], b[0]], [a[1], b[1]], [tnear, tnear + ti]
                G['vertex'].append(Qcurr)
                G['edge'][str(counter)] = path_curr
                G['neighbor'][str(counter)] = [(a[0], a[1], tnear), (b[0], b[1], tnear + ti)]
                G['t'].append(tnear + ti)

            # Update termination conditoin
                dist_curr = self.get_range(Qcurr, xf)
                if dist_curr < dist_best:
                    dist_best = dist_curr
                    Qnear_prev = Qcurr
                    tprev = tnear + ti

            # If within tolerance, stop and compute path
                if dist_best < dist_tol:
                    print('Destination Reached')
                    # Stop iteration
                    repeat_R2T2 = False

                    # Find route to Final Point
                    route = self.rrt_course(G)
                    G['route'] = route
                    route_tvec = []
                    for ri in np.flip(route):
                        route_tvec.append(G['t'][ri])
                    route_tvec.append(tnext)

                    return G
            
            # If not reached, update for next loop
            counter += 1
            
            # Break out if loop is too long
            if np.abs(G['t'][-1] - self.endtime) <= 1e-2 or counter > 150:
                print('Fail to reach destination in time')
                return G
                

        
    # Functiosn for Step 3
    def get_range(self, X0, X1):
        diff = X1 - X0
        print(diff)
        return np.sqrt((diff[0])**2 + (diff[1])**2 + (diff[2])**2)

    def gen_node(self, r, qnear, t_in):
        # Generate qrand in polar coordinate and convert to cartesian
        qrand_r = rn.uniform(0, r)
        qrand_th = rn.uniform(0, 2*np.pi)
        qrand_x = qrand_r*np.cos(qrand_th)
        qrand_y = qrand_r*np.sin(qrand_th)
        qrand = np.array([qrand_x+qnear[0], qrand_y+qnear[1], qrand_th])

        # Check if random sample is valid
        if self.static_bound(qrand) and (not qrand is None) and not self.dynamic_bound(qrand, t_in):
            return qrand
        else:
            return self.gen_node(r, qnear, t_in)

    def static_bound(self, xi):
        # Check if random sample's validity
        # True:  It is valid
        # False: It is not valid
        return self.in_map(xi) and not self.building_bound(xi)
        
    def in_map(self, xi):
        # Check if point is in map
        # True:  Point in map
        # False: Point outside of 
        return (xi[0] >= self.map_st['size'][0] and xi[0]<=self.map_st['size'][1] and xi[1] >= self.map_st['size'][2] and xi[1]<=self.map_st['size'][3] and xi[2] >= self.map_st['size'][4] and xi[2]<=self.map_st['size'][5])

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
            vert_avail = VERT_avail
            temp = G['t'][ii]
            print(xi, vert_avail)
            range_vec.append(np.sqrt((xi[0]-vert_avail[0])**2 + (xi[1]-vert_avail[1])**2 + (xi[2]-vert_avail[2])**2))
        Xnear = G['vertex'][range_vec.index(min(range_vec))]
        t_at_Xnear = G['t'][range_vec.index(min(range_vec))]

        return Xnear, t_at_Xnear
    
    # Functions for Step 5
    def local_path_planner(self, X0, X1, dist):
        # Plot straight route
        time_taken = dist / self.vmax
        
        return np.linspace(X0, X1, time_taken)

    def collision_check(self, Q0, Q1, qtvec, steps):
        test_vector = np.linspace(Q0, Q1, steps)
        
        for tt in test_vector:
            if not self.in_map(tt):
                if bool(self.building_bound(tt)):
                    return True
            
            print(test_vector == t)
            afdasfdafs
            
            t_in_to_test, = np.where(test_vector == tt)
            t_to_test = qtvec[t_in_to_test[0]]

            if bool(self.dynamic_bound(tt, t_to_test)):
                return True
            
        return False
    
    def current_pos(self, Q0, Q1, V, qt_vec):
        v = SE2.log(SE2(SE2.to_vec(Q0.inv@V.to_matrix)))
        curr_path = {}
        curr_path['x'] = []
        curr_path['y'] = []
        curr_path['t'] = []
        curr_path['theta'] = []
        tscale = qt_vec[-1] - qt_vec[0]

        for tt in np.linspace(0, 1, len(qt_vec)):
            Q = Q0@SE2.exp(v.rmul(tt))
            xt = SE2.to_vec(Q)[0]
            yt = SE2.to_vec(Q)[1]
            thetat = SE2.to_vec(Q)[2]
            curr_path['x'].append(xt)
            curr_path['y'].append(yt)
            curr_path['t'].append(qt_vec[0]+tscale*tt)
            curr_path['theta'].append(thetat)

            if self.collision_check(self.vradius, Q0, SE2(SE2.to_vec(Q)), [tt], 1):
                return SE2(np.array([xt, yt, thetat])), curr_path
            
        return SE2(np.array([xt, yt, thetat])), curr_path



    # Functions for Step 7
    def rrt_course(self, G):
        print('Plotting Course')
        neighbor = G['neighbor']
        key_list = neighbor.keys()

        # Route Index
        route = []
        
        def match_vertex(v0, v1):
            tol = 1e-2
            if np.abs(v0[0]-v1[0]) < tol and np.abs(v0[1]-v1[1]) < tol:
                return True
            else:
                return False
            
        # Start from initial point
        q0 = self.Xinit
        qf = self.Xfin
        qcheck = qf

        # Find matching vertices
        castellan = True
        while castellan:
            print('Awakening the Machine Spirit...')
            for ii in range(len(key_list)):
                # Extract iith key
                currkey = list(key_list)[ii]
                if match_vertex(qcheck, neighbor[str(currkey)][1]):
                    route.append(ii)
                    qcheck = neighbor[str(currkey)][0]
                    # Return route to final point
                    if match_vertex(qcheck, q0):
                        castellan = False
                        return route




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
    vehicle['yaw'] = np.deg2rad(30)
    vehicle['pitch'] = np.deg2rad(30)

    # Map
    map_in = {}
    # Static Map
    map_in['st'] = {}
    map_in['st']['size'] = np.array([-5, 15, -15, 15, 0, 10])
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
    cam_z = np.array([10])
    cam_dict = {}
    cam_dict['n'] = len(cam_x)
    cam_dict['x'] = cam_x
    cam_dict['y'] = cam_y
    cam_dict['z'] = cam_y


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