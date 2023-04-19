import numpy as np
import matplotlib.pyplot as plt
import random as rn
import shapely
from shapely import geometry

# 3D Plot
from mpl_toolkits import mplot3d

# Custom Classes
from SE2 import SE2, se2
from Camera import Camera
from Dynamic import DynamicMap

# Global functions
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
    return np.array([u[0], u[1], theta])

def SE2_exp(v):
    """
    SE2 matrix exponential
    """
    x, y, theta = v
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

def SE2_to_param(M):
    """
    From matrix to [theta, x, y]
    """
    theta = np.arctan2(M[1, 0], M[0, 0])
    x = M[0, 2]
    y = M[1, 2]
    return np.array([x, y, theta])


# R2T2
class R2T2:
    """
    RRT with Time Domain

    TODO
    1. Complete 2D RRT
    1.1 Build RRT with SE2
    2. Complete 3D RRT
    """
    
    def __init__(self, x0, x1, t, vehicle, map_in):
        # x: 2x1 vector, [xinit, xfin]
        xi_x = x0[0]
        xi_y = x0[1]
        self.thet_init = x0[2]
        self.xinit = (xi_x, xi_y, self.thet_init)
        xf_x = x1[0]
        xf_y = x1[1]
        self.thet_fin = x1[2]
        self.xfin = (xf_x, xf_y, self.thet_fin)
        
        # t: 3x1 vector, [trange, delt, endtime]
        self.trange = t[0]
        self.delt = t[1]
        self.endtime = t[2]
        
        # vehcile: dictionary
        # key: v
        # 'v' = max velocity
        # 'radius' = radius
        self.vmax = vehicle['v']
        self.vrad = vehicle['radius']

        # map: dictionary
        # key: n, st, dy, curr
        # 'n' = repeat period length
        # 'ncam' = number of camera present
        # 'st' = static map
        # ['st']['size'] = 4x1 vector [x0, xmax, y0, ymax]
        # ['st']['n'] = number of buildings
        # ['st'][str(i)], i = number // building poly  at ith index
        # 'dy' = repeating dynamic map with n periods
        # ['dy'][str(i)], i = number // dynamic map at ith index
        # 'curr' = static + dynamic = current map
        # ['curr'][str(i)], i = number = currnet map at ith index
        self.map_st = map_in['st']
        self.map_dy = map_in['dy']
        #self.map_period = map_in['n']
        #self.map_curr = map_in['curr']

    
    def R2T2_2D(self):
        """
        This a 2D RRT with time domain as z axis
        1. Generate Random t
        2. Find range of r it can reach
        3. Generate random coordinate x1, and check whether or not it is located inside the cone, and find closest qnear // Or can we select any number within cone volume?
           80% to move to new qrand
           10% to stay
           10% to move to final point
        4. IF x1 is located inside code, connect it with start point x0 // IF x1 NOT located in cone, regen random coordinate
        5. Can there be path generated?
        6. Repeat 3-4 until cone include final point
        7. Connect from xfin to xinit and generate path

        TODO
        - Convert plt to 3D, z axis being time domain
        """
        # Initial Variables
        xi = self.xinit
        xf = self.xfin
        dt = self.delt
        trange = self.trange

        # Set Emptry Storages
        # Vertex:
        # Edge: 
        # t: 
        G = {}
        G['vertex'] = []
        G['neighbor'] = {}
        G['edge'] = {}
        G['route'] = []
        G['t'] = []
        
        # Initialize R2T2
        Xi = SE2(x=xi[0], y=xi[1], theta=xi[2])
        Xf = SE2(x=xf[0], y=xf[1], theta=xf[2])
        Qnear = SE2(x=xi[0], y=xi[1], theta=xi[2])
        G['vertex'].append(Qnear)
        G['t'].append(0)
        Qnear_prev = G['vertex'][-1]
        tprev = 0
        map_size = self.map_st['size']
        dmap = self.map_dy
        bad_path = []

        # Start loop, and repeat until route toward final point
        repeat_R2T2 = True
        plot = True
        debug = False
        with2D = True
        counter = 0

        # End condition
        dist_best = self.get_range(Xi, Xf)
        dist_tol = 1e-1

        if plot:
            fig = plt.figure(figsize=(7.5,7.5))
            #ax = plt.axes(projection ='3d')
            ax = fig.add_subplot(111,projection='3d')
            
            # Graph building
            for i in range(self.map_st['n']):
                ibuilding = self.map_st[str(i)]
                wall = geometry.LineString(ibuilding)
                building = geometry.Polygon(wall)
                bx,by = building.exterior.xy
                plt.plot(bx, by, 0, '-k', label = 'Building')

            # Graph camera at t0
            cameras_at_t0 = dmap.gen_cam(G['t'][-1])
            for i in range(cameras_at_t0['n']):
                camera_i_FOV = cameras_at_t0[str(i)]['FOV']
                camera_i_FOV_Poly = cameras_at_t0[str(i)]['FOV_Poly']
                # Plot Camera Location
                plt.plot(camera_i_FOV[0,0], camera_i_FOV[0,1], 'og', label = 'Camera')
                # Plot FOV Polygon
                FOV_Poly_xi_t0, FOV_Poly_yi_t0 = camera_i_FOV_Poly.exterior.xy
                plt.plot(FOV_Poly_xi_t0, FOV_Poly_yi_t0, '-g')

        while repeat_R2T2:
            # Step 0
            # Setup
            #Qnear_prev = G['vertex'][-1]
            
            print('=============================')
            print('counter: ', counter)
            # Step 1
            # Generate random time t to move
            ti = rn.uniform(0,trange)

            # Step 2
            # Compute possible movement range
            mv_R = self.vmax*ti

            # Step 3
            # Choose next move qnext
            # 80% -> move to random point qrand
            # 10% -> remain stationary
            # 10% -> route to final point
            if not debug:
                switch = [.8, .1, .1]
                Qnear = None
                Qnear_exist = False
                print(bool(Qnear_exist))
                while not Qnear_exist:
                    if Qnear is None:
                        choose = rn.uniform(0,1)
                        if choose >= 0 and choose <= switch[0]:
                            print('Random')
                            # Get random point x1, and check whether or not it is valid
                            Qnext = self.gen_node(mv_R, Qnear_prev.SE2param(), 0, tprev)
                            Qnear_exist = True
                        elif choose > switch[0] and choose <= switch[2]:
                            print('Stationary')
                            # Statinoary
                            # TODO
                            # Make sure the point is still valid from ti seconds from now                        
                            Qnext = Qnear_prev
                            Qnear_exist = True
                        else:
                            print('Final')
                            Qnext = SE2(xf[0], xf[1], xf[2])
                            Qnear_exist = True
                    print(bool(Qnear_exist))

                # Step 4
                # Find cloest point from qnear
                Qnear, tnear = self.nearest(G, Qnext.SE2param())
                tnext = tnear+ti

            """
            Debug tool
            if debug:
                if counter == 0:
                    Qnear = SE2(0,0,np.pi/2)
                    Qnext = SE2(0,3,np.pi/2)#SE2(3.724169139935386, 6.702351939209903, 4.93572416536846)
                    ti = 7.667526152539541
                    #Qnear = SE2(0, 5, np.pi/2)
                    #Qnext = SE2(2, 8, 0)
                    ti = 10
                    tnear = 0
                elif counter == 1:
                    Qnear = Qcurr#SE2(3.4403697706430325, 6.5172588623971555, 0.5993828445723643)
                    Qnext = SE2(0, 6, np.pi/2)
                    ti = 3.6185084339182403
                    tnear = tnext#7.667526152539541
                elif counter == 2:
                    Qnear = Qcurr
                    Qnext = SE2(10, 7, 0)
                    ti = 10
                    tnear = tnext
                elif counter == 3:
                    Qnear = Qcurr
                    Qnext = SE2(10, 2, -np.pi/2)
                    ti = 10
                    tnear = tnext
                elif counter == 4:
                    Qnear = SE2(0, 6, np.pi/2)
                    Qnext = SE2(-1, 4, -np.pi/2)
                    ti = 5
                    tnear = tnext
                tnext = tnear+ti
            """
            print('Qnear ', Qnear.SE2param())
            print('Qnext ', Qnext.SE2param())
            print('trange ', trange)
            print('ti ', ti)
            print('tnear ', tnear)
            print('tnext ', tnext)
            st_tvec = np.arange(tnear, tnext, self.delt)

            # Step 5
            # Generate Path and check collision
            # This is done in SE2 Lie Group for 2D R2T2
            Qroute = self.local_path_planner(Qnear, Qnext, self.vmax*ti)

            # Check Collision
            # If collision occur, we ignore that path
            # If collision doesn't occur, carry on
            if self.collision_check(self.vrad, Qnear, Qroute, st_tvec, len(st_tvec)):
                bad_path.append(Qnext)
                print('Route Collision Detected')
            #    print('Route: ', SE2_to_param(Qroute))
            else:
                # Compute current position as time may differ
                Qcurr, path_curr = self.current_pos(Qnear, Qnext, Qroute, st_tvec)
                print('Qcurr ', Qcurr.SE2param())

                # TODO
                # Dynamic Test Check
                if debug:
                    print('==========================')
                    if counter == 3:
                        x_to_test = path_curr['x']
                        y_to_test = path_curr['y']
                        t_to_test = path_curr['t']
                        for checking_index in range(len(t_to_test)):
                            print(x_to_test[checking_index], y_to_test[checking_index], t_to_test[checking_index])
                            xkek = x_to_test[checking_index]
                            ykek = y_to_test[checking_index]
                            tkek = t_to_test[checking_index]
                            print(bool(self.dynamic_bound((xkek, ykek), tkek)))
                    print('==========================')


                if plot:
                    # 3D Plot
                    # Plot Qnear
                    qrpt = Qnear.SE2param()
                    plt.plot(qrpt[0], qrpt[1], tnear, '.r', alpha=.5, label='Nearest vertex', zorder=15)

                    # Plot Qnext
                    qnpt = Qnext.SE2param()
                    plt.plot(qnpt[0], qnpt[1], tnext, 'xr', alpha = .5, label='Next vertex', zorder=15)

                    # Plot Qcurr
                    qcpt = Qcurr.SE2param()
                    plt.plot(qcpt[0], qcpt[1], tnext, '+r', label='Current position reachable at t', zorder=15)
                    
                    # Plot Current Path
                    plt.plot(path_curr['x'], path_curr['y'], path_curr['t'],'-r', linewidth=5, alpha=.25, label='Computed route', zorder=15)

                    # 2D Plot
                    if with2D:
                        # Plot Qnear
                        plt.plot(qrpt[0], qrpt[1], '.b', alpha=.5, label='Nearest Vertex in 2D', zorder=15)

                        # Plot Qnext
                        plt.plot(qnpt[0], qnpt[1], 'xb', alpha = .5, label='Next vertex in 2D', zorder=15)

                        # Plot Qcurr
                        plt.plot(qcpt[0], qcpt[1], '+b', label='Current position reachable at t in 2D', zorder=15)

                        # Plot Projection
                        plt.plot([qcpt[0], qcpt[0]], [qcpt[1], qcpt[1]], [0, tnext], '--k', alpha=0.1, zorder=15)
                        
                        # Plot Current Path
                        plt.plot(path_curr['x'], path_curr['y'], '-b', linewidth=5, alpha=.25, label='Comptued route in 2D', zorder=15)
                    
                    
                    if debug and counter == 3:
                        a = Qnear.SE2param()
                        b = Qcurr.SE2param()
                        edge = [a[0], b[0]], [a[1], b[1]], [tnear, tnear+ti]
                        G['vertex'].append(Qcurr)
                        G['edge'][str(counter)] = path_curr
                        G['neighbor'][str(counter)] = [(a[0], a[1], tnear), (b[0], b[1], tnear + ti)]
                        G['t'].append(tnear + ti)

                        # Plot Static Map
                        R2T2_tvec = np.arange(0, G['t'][-1], self.delt)
                        for i in range(self.map_st['n']):
                            ibuilding = self.map_st[str(i)]
                            wall = geometry.LineString(ibuilding)
                            building = geometry.Polygon(wall)
                            bx,by = building.exterior.xy
                            for st_i in R2T2_tvec:
                                plt.plot(bx, by, st_i, '-k', alpha=0.1)
                        plt.plot(bx, by, st_i, '-k', alpha=1)

                        # Plot Dynamic Map
                        dmap = self.map_dy
                        for st_i in R2T2_tvec:
                            cameras = dmap.gen_cam(st_i)
                            ncam = cameras['n']
                            for ii in range(ncam):
                                cam_i = cameras[str(ii)]['FOV_Poly']
                                cx, cy = cam_i.exterior.xy
                                plt.plot(cx, cy, st_i, '-g', alpha=0.1)
                        plt.plot(cx, cy, st_i, '-g', alpha=1)

                        plt.xlim([map_size[0], map_size[1]])
                        plt.ylim([map_size[2], map_size[3]])
                        #plt.legend()
                        plt.ioff() # Interactice Plot Trigger
                        plt.show()
                    

            # Step 6
            # Check if destination is reached
            # Update Dictionary

                a = Qnear.SE2param()
                b = Qcurr.SE2param()
                edge = [a[0], b[0]], [a[1], b[1]], [tnear, tnear+ti]
                G['vertex'].append(Qcurr)
                G['edge'][str(counter)] = path_curr
                G['neighbor'][str(counter)] = [(a[0], a[1], tnear), (b[0], b[1], tnear + ti)]
                G['t'].append(tnear + ti)

            # Update termination condition
            #dist_curr = self.get_range(Qcurr, Xf)
                dist_curr = self.get_range(Qcurr, Xf)
                print('curr', dist_curr)
                print('best', dist_best)
                if dist_curr < dist_best:
                    dist_best = dist_curr
                    Qnear_prev = Qcurr
                    tprev = tnear+ti

            # If within tolerance, we stop and return path
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

                    if plot:
                        # Plot route
                        cawl = G['edge']
                        cawl_key = list(cawl.keys())
                        cawl_counter = 0
                        for ri in np.flip(route[0:len(route)]):
                            ed = cawl[str(cawl_key[ri])]
                            #plt.plot(ed['x'], ed['y'], np.linspace(route_tvec[cawl_counter], route_tvec[cawl_counter+1], len(ed['x'])), '--r')
                            plt.plot(ed['x'], ed['y'], ed['t'], '--r', zorder=10)
                            if with2D:
                                plt.plot(ed['x'], ed['y'], '--b', zorder=10)
                            cawl_counter += 1

                        # Plot Static Map
                        R2T2_tvec = np.arange(0, G['t'][-1], self.delt)
                        for i in range(self.map_st['n']):
                            ibuilding = self.map_st[str(i)]
                            wall = geometry.LineString(ibuilding)
                            building = geometry.Polygon(wall)
                            bx,by = building.exterior.xy
                            for st_i in R2T2_tvec:
                                # Currently set alpha = 0
                                plt.plot(bx, by, st_i, '-k', alpha=0)
                        for i in range(5):
                            plt.plot([bx[i], bx[i]], [by[i], by[i]], [0, st_i], '-k')
                        plt.plot(bx, by, st_i, '-k', alpha=1)

                        # Plot Dynamic Map
                        dmap = self.map_dy
                        for st_i in R2T2_tvec:
                            cameras = dmap.gen_cam(st_i)
                            ncam = cameras['n']
                            for ii in range(ncam):
                                cam_i = cameras[str(ii)]['FOV_Poly']
                                cx, cy = cam_i.exterior.xy
                                plt.plot(cx, cy, st_i, '-g', alpha=0.1, zorder=5)
                        plt.plot(cx, cy, st_i, '-g', alpha=1, zorder=5)

                        # Plot xi, xf
                        if with2D:
                            plt.plot(xi[0], xi[1], 'xr')
                            plt.plot(xf[0], xf[1], 'xb')
                        plt.plot(xi[0], xi[1], 0, '^r')
                        plt.plot(xf[0], xf[1], G['t'][-1], '^b')

                        # Plot Settings
                        plt.xlim([map_size[0], map_size[1]])
                        plt.ylim([map_size[2], map_size[3]])
                        #plt.legend()
                        ax.set_xlabel('X [m]')
                        ax.set_ylabel('Y [m]')
                        ax.set_zlabel('t [sec]')
                        plt.ioff() # Interactice Plot Trigger
                        plt.grid()
                        plt.show()
                    return G

            # If not, we update for next loop
            print('\n')
            counter += 1

            # Break out if loop is too long
            if np.abs(G['t'][-1]-self.endtime) <= 1e-2 or counter > 150:
                print('Fail to reach destination in time')

                if plot and not debug:
                    # Plot Static Map
                    end_time_tvec = np.max(G['t'])
                    R2T2_tvec = np.arange(0, tnext+ti, self.delt)
                    for i in range(self.map_st['n']):
                        ibuilding = self.map_st[str(i)]
                        wall = geometry.LineString(ibuilding)
                        building = geometry.Polygon(wall)
                        bx,by = building.exterior.xy
                        for st_i in R2T2_tvec:
                            plt.plot(bx, by, st_i, '-k', alpha=0.1)
                    plt.plot(bx, by, end_time_tvec, '-k', alpha=1)

                    # Plot Dynamic Map
                    dmap = self.map_dy
                    for st_i in R2T2_tvec:
                        cameras = dmap.gen_cam(st_i)
                        ncam = cameras['n']
                        for ii in range(ncam):
                            cam_i = cameras[str(ii)]['FOV_Poly']
                            cx, cy = cam_i.exterior.xy
                            plt.plot(cx, cy, st_i, '-g', alpha=0.1)
                    plt.plot(cx, cy, end_time_tvec, '-g', alpha=1)

                    # Plot xi, xf
                    if with2D:
                        plt.plot(xi[0], xi[1], 'xr')
                        plt.plot(xf[0], xf[1], 'xb')
                    plt.plot(xi[0], xi[1], 0, '^r')
                    plt.plot(xf[0], xf[1], tnext+ti, '^b')

                    # Plot Settings
                    plt.xlim([map_size[0], map_size[1]])
                    plt.ylim([map_size[2], map_size[3]])
                    #plt.legend()
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('t [sec]')
                    plt.ioff() # Interactice Plot Trigger
                    plt.grid()
                    plt.show()  
                return G


    # Functios for Step 3
    def gen_node(self, r, qnear, c, t_in):
        count = c
        # Generate qrand
        qrand_x = rn.uniform(-r, r)
        qrand_y = np.sqrt(r**2 - qrand_x**2)*np.sign(rn.uniform(-r, r))
        #qrand_y = rn.uniform(-r, r)
        qrand_th = 2*np.pi*rn.uniform(0,1)
        qrand = (qrand_x + qnear[0], qrand_y + qnear[1], qrand_th)
        Qrand_SE2 = SE2(qrand[0], qrand[1], qrand[2])

        # check 2 things:
        # 1. qrand is within movement bound
        # 2. qrand is not in static map

        # With current setup, always assume max V,
        # So it will always in movement bound
        #if self.move_bound(r, [qrand_x, qrand_y]) and not self.static_bound(qrand):
        if c <= 100:
            if self.static_bound(qrand) and not Qrand_SE2 is None and not self.dynamic_bound(qrand, t_in):
                return Qrand_SE2
            else:
                count+=1
                return self.gen_node(r,qnear, count, t_in)
        else:
            # If loop is too much, just return stationary
            return SE2(qnear[0], qnear[1], qnear[2])
        
    def building_bound(self, xi):
        # Check if point is inside building polygon
        smap = self.map_st
        check_vec = []
        for i in range(smap['n']):
            # ibuilding is a list of vertex of building polygon
            ibuilding = smap[str(i)]
            wall = geometry.LineString(ibuilding)
            building = geometry.Polygon(wall)

            check_vec.append(building.contains(geometry.Point(xi[0],xi[1])))

        # Add all boolean values,
        # if 1: Point inside building
        # if 0: Point not contained within building
        return np.sum(check_vec)

    def move_bound(self, r, xi):
        # Is xi within possible movement range in given time?
        return bool(xi[0]**2+xi[1]**2 <= r)

    def in_map(self, xi):
        # Check if ponit is within map
        return (xi[0] >= self.map_st['size'][0] and xi[0]<=self.map_st['size'][1] and xi[1] >= self.map_st['size'][2] and xi[1]<=self.map_st['size'][3])

    def static_bound(self, xi):
        return self.in_map(xi) and not self.building_bound(xi)
    
    def dynamic_bound(self, xi, t_in):
        """
        1. At given time input t_in, obtain dynamic masking.
        2. Then, check whether or not that given piont is inside any camera FOV
        Input: t_in = time index (continuous)
        Output: Dynamic masking at that point
        """
        # Continuous Dynamic Map function is generated during setup phase
        dmap = self.map_dy

        # Extract camera function dictionary
        # Camera function has following keys
        # ['n'] = number of camera
        # [str(i)]['Camera'] = ith camera object
        # [str(i)]['FOV'] = Computed camera FOV triangle vertices
        # [str(i)]['FOV_Poly'] = Computed polygon object from FOV vertices
        cameras = dmap.gen_cam(t_in)
        ncam = cameras['n']
        check_vec = []
        for i in range(ncam):
            # Extract FOV Polynomial of ith camera
            cam_i = cameras[str(i)]['FOV_Poly']
            check_vec.append(cam_i.contains(geometry.Point(xi[0],xi[1])))
        
        # Add all boolean values,
        # if 1: Point inside any camera FOVs
        # if 0: Point is outside of all camera FOVs
        return np.sum(check_vec)

    # Functions for Step 4
    def nearest(self, G, xi):
        3# Find nearest vertex
        range_vec = []
        for ii in range(len(G['vertex'])):
            VERT_avail = G['vertex'][ii]
            vert_avail = VERT_avail.SE2param()
            temp = G['t'][ii]
            range_vec.append(np.sqrt((xi[0] - vert_avail[0])**2 + (xi[1] - vert_avail[1])**2))

        Xnear = G['vertex'][range_vec.index(min(range_vec))]
        t_at_Xnear = G['t'][range_vec.index(min(range_vec))]

        return Xnear, t_at_Xnear
    
    # Functions for Step 5
    def find_u_R_d(self, X0, X1):
        # Compute arc length, radius, distance
        M = X0.inv()@X1.matrix()
        dx, dy, dth = SE2_to_param(M)

        d = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy, dx)

        if np.abs(alpha) > 1e-3:
            R = d/(2*np.sin(alpha))
            u = 2*R*alpha
        else:
            R = np.infty
            u = d

        return u, R, d

    def local_path_planner(self, X0, X1, dist):
        # Plot route from start X0 to end X1
        u, R, d = self.find_u_R_d(X0, X1)
        if np.abs(u) > dist:
            u = dist*np.sign(u)

        # Compute turn angle omega
        if np.isclose(np.abs(R),0):
            omega = 0
        else:
            omega = u/R

        v = se2(u, 0, omega)
        V = v.exp()

        return X0.matrix()@V
    
    def current_pos(self, Q0, Q1, V, qt_vec):
        """
        TODO
        If current_pos is outside of bounds
        -> return to position where it is valid
        """
        v = SE2_log(Q0.inv()@V)
        curr_path = {}
        curr_path['x'] = []
        curr_path['y'] = []
        curr_path['t'] = []
        curr_path['theta'] = []
        tscale = qt_vec[-1]-qt_vec[0]

        #for tt in np.arange(0,1,self.delt):
        for tt in np.linspace(0, 1, len(qt_vec)):
        #for tt in np.linspace(0,qt_vec[-1]-qt_vec[0],len(qt_vec)):
            Q = Q0.matrix()@(SE2_exp(v*tt))
            xt, yt, thetat = SE2_to_param(Q)
            curr_path['x'].append(xt)
            curr_path['y'].append(yt)
            curr_path['t'].append(qt_vec[0]+tscale*tt)
            curr_path['theta'].append(thetat)

            # Check if current position is on collision point
            if self.collision_check(self.vrad, Q0, Q, [tt], 1):
                return SE2(x=xt, y=yt, theta=thetat), curr_path
            
        return SE2(x=xt, y=yt, theta=thetat), curr_path
    
    def collision_check(self, vehicle_radius, Q0, Q1, qtvec, steps):
        """
        TODO
        Add collision for dynamic case
        """
        V = Q0.inv()@(Q1)
        v = SE2_log(V)
        test_vector = np.linspace(0,1,steps)
        for tt in test_vector:
            Q = Q0.matrix()@(SE2_exp(v*tt))
            xt, yt, thetat = SE2_to_param(Q)
            
            # check map bounds
            if not self.in_map([xt, yt, thetat]):
                return True

            # check collision with static obstacles
            if self.in_map([xt, yt, thetat]):                
                if bool(self.building_bound([xt, yt, thetat])):
                    return True
            # check collision with dynamic obstacles at matching time in time vector qtvec
            t_in_to_test, = np.where(test_vector == tt)
            t_to_test = qtvec[t_in_to_test[0]]
            if bool(self.dynamic_bound((xt, yt), t_to_test)):
                return True
        return False
    
    # Functions for Step 6
    def get_range(self, Q, Xf):
        diff = Xf.matrix() - Q.matrix()
        return np.sqrt((diff[0,2])**2 + (diff[1,2])**2)
    
    # Functions for Step 7
    def rrt_course(self, G):
        print('Plotting Course')
        neighbor = G['neighbor']
        key_list = neighbor.keys()

        # Route Index
        route = []

        def match_vertex(v0, v1):
            tol = 1e-2
            if np.abs(v0[0]-v1[0])<tol and np.abs(v0[1]-v1[1])<tol:
                return True
            else:
                return False

        # Start from initial point
        # points in se3
        q0 = self.xinit
        qf = self.xfin
        qcheck = qf

        # Find matching vertices
        castellan = True
        while castellan:
            print('Blessing the Machine...')
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
if __name__ == "__main__":
    import numpy as np
    from Camera import Camera
    from Dynamic import DynamicMap

    # Test case
    # Initial, Final Positions
    x0 = np.array([0, 0, 0])
    x1 = np.array([10, 1, 0])
    t = [20, .1, 100]

    # Vehicle Spec
    vehicle = {}
    vehicle['v'] = 1
    vehicle['radius'] = 0.5

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
    tilt_limit = np.array([np.pi/2, 0]) #[upper, lower]
    fov_ang = np.deg2rad(20)
    fov_rng = 5 #[m]
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
#%%