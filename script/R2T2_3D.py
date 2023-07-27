import numpy as np
import matplotlib.pyplot as plt
import random as rn
import shapely
from shapely import geometry

# 3D Plot
from mpl_toolkits import mplot3d

# Custom Classes
from Camera_3D import Camera
from Dynamic_3D import DynamicMap

# R2T2
class R2T2:
    """
    RRT with Time Domain
    """
    
    def __init__(self, x0, x1, t, vehicle, map_in):
        # x: 3x1 vector, [x, y, z, phi, theta, psi]
        xi_x = x0[0]
        xi_y = x0[1]
        xi_z = x0[2]
        self.phi_init = x0[3]
        self.thet_init = x0[4]
        self.psi_init = x0[5]
        self.xinit = (xi_x, xi_y, xi_z, self.phi_init, self.thet_init, self.psi_init)
        xf_x = x1[0]
        xf_y = x1[1]
        xf_z = x1[2]
        self.phi_fin = x1[3]
        self.thet_fin = x1[4]
        self.psi_fin = x1[5]
        self.xfin = (xf_x, xf_y, xf_z, self.phi_fin, self.thet_fin, self.psi_fin)
        
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
        self.vang = vehicle['w']

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

    
    def R2T2_3D(self):
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
        tf = self.endtime

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
        Xi = np.array([xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]])
        Xf = np.array([xf[0],xf[1],xf[2],xf[3],xf[4],xf[5]])

        # Check if of Xf at tf is good:
        if not self.static_bound(xf) or Xf is None: # or self.dynamic_bound(xf, self.endtime): TODO to fix
            raise Exception('Invalid Final Point Xf at tf')

        Qnear = Xi
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
            # 3 Axis: XYZ
            fig = plt.figure(figsize=(7.5,7.5))
            #ax = plt.axes(projection ='3d')
            ax = fig.add_subplot(111,projection='3d')
            
            # Graph building
            for i in range(self.map_st['n']):
                ibuilding = self.map_st[str(i)]
                wall = geometry.LineString(ibuilding)
                building = geometry.Polygon(wall)
                bx,by = building.exterior.xy
                bz = ibuilding[0][2]
                plt.plot(bx, by, bz, '-k', label = 'Building')

            # Graph camera at t0
            cameras_at_t0 = dmap.gen_cam(G['t'][-1])
            for i in range(cameras_at_t0['n']):
                camera_i_FOV = cameras_at_t0[str(i)]['FOV']
                #camera_i_FOV_Poly = cameras_at_t0[str(i)]['FOV_Poly']
                # Plot Camera Location
                plt.plot(camera_i_FOV[0][0], camera_i_FOV[0][1], camera_i_FOV[0][2], 'og', label = 'Camera')
                # Plot FOV Polygon
                #FOV_Poly_xi_t0, FOV_Poly_yi_t0 = camera_i_FOV_Poly.exterior.xy
                #plt.plot(FOV_Poly_xi_t0, FOV_Poly_yi_t0, '-g')

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
                while not Qnear_exist:
                    if Qnear is None:
                        choose = rn.uniform(0,1)
                        if choose >= 0 and choose <= switch[0]:
                            print('Random')
                            # Get random point x1, and check whether or not it is valid
                            Qnext = self.gen_node(mv_R, Qnear_prev, 0, tprev)
                            Qnear_exist = True
                        elif choose > switch[0] and choose <= switch[0]+switch[1]:
                            print('Stationary')
                            # Statinoary
                            # TODO
                            # Make sure the point is still valid from ti seconds from now                        
                            Qnext = Qnear_prev
                            Qnear_exist = True
                        else:
                            print('Final')
                            Qnext = Xf
                            Qnear_exist = True
                    print(bool(Qnear_exist))
                print(Qnext)

                # Step 4
                # Find cloest point from qnear
                Qnear, tnear = self.nearest(G, SE3.to_vec(Qnext.to_matrix))
                tnext = tnear+ti
            
            print('Qnear ', Qnear)
            print('Qnext ', Qnext)
            print('trange ', trange)
            print('ti ', ti)
            print('tnear ', tnear)
            print('tnext ', tnext)
            st_tvec = np.arange(tnear, tnext, self.delt)

            # Step 5
            # Generate Path and check collision
            # This is done in SE2 Lie Group for 2D R2T2
            Qroute = self.local_path_planner(Qnear, Qnext, self.vmax*ti)

            adsfasdfaf

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

                if plot:
                    # 3D Plot
                    # Plot Qnear
                    qrpt = Qnear.SE2param()
                    plt.plot(qrpt[0], qrpt[1], tnear, '.r', alpha=.5, label='Nearest vertex', zorder=15)

                    # Plot Cone
                    cone_n = 20
                    r_vec = np.linspace(0, ti*self.vmax, cone_n)
                    cone_t_vec = np.linspace(tnear, tnext, cone_n)
                    ang_vec = np.linspace(0, np.pi*2, 20)
                    for i in range(cone_n):
                        cone_x = r_vec[i]*np.cos(ang_vec)+qrpt[0]
                        cone_y = r_vec[i]*np.sin(ang_vec)+qrpt[1]
                        plt.plot(cone_x, cone_y, cone_t_vec[i], '-m', alpha=0.15)

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

                        # Plot Reverse cone
                        cone_n = 100
                        r_vec = np.linspace(self.endtime*self.vmax, 0, cone_n)
                        cone_t_vec = np.linspace(0, self.endtime, cone_n)
                        ang_vec = np.linspace(0, np.pi*2, 20)
                        for i in range(cone_n):
                            cone_x = r_vec[i]*np.cos(ang_vec)+xf[0]
                            cone_y = r_vec[i]*np.sin(ang_vec)+xf[1]
                            plt.plot(cone_x, cone_y, cone_t_vec[i], '-c', alpha=0.15)

                        # Plot xi, xf
                        if with2D:
                            plt.plot(xi[0], xi[1], 'xr')
                            plt.plot(xf[0], xf[1], 'xb')
                        plt.plot(xi[0], xi[1], 0, '^r')
                        plt.plot(xf[0], xf[1], self.endtime, '^b')

                        # Plot Settings
                        plt.xlim([map_size[0], map_size[1]])
                        plt.ylim([map_size[2], map_size[3]])
                        #plt.legend()
                        ax.set_xlabel('X [m]')
                        ax.set_ylabel('Y [m]')
                        ax.set_zlabel('t [sec]')
                        plt.ioff() # Interactice Plot Trigger
                        plt.axis('equal')
                        plt.grid()
                        plt.show()
                    return G

            # If not, we update for next loop
            print('\n')
            counter += 1

            # Break out if loop is too long
            if np.abs(G['t'][-1]-self.endtime) <= 1e-2 or counter > 100:
                print('Fail to reach destination in time')

                if plot and not debug:
                    # Plot Static Map
                    R2T2_tvec = np.arange(0, self.endtime, self.delt)
                    for i in range(5):
                        plt.plot([bx[i], bx[i]], [by[i], by[i]], [0, self.endtime], '-k')
                    plt.plot(bx, by, self.endtime, '-k', alpha=1)

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
                    plt.plot(xf[0], xf[1], self.endtime, '^b')

                    # Plot Reverse cone
                    cone_n = 100
                    r_vec = np.linspace(self.endtime*self.vmax, 0, cone_n)
                    cone_t_vec = np.linspace(0, self.endtime, cone_n)
                    ang_vec = np.linspace(0, np.pi*2, 20)
                    for i in range(cone_n):
                        cone_x = r_vec[i]*np.cos(ang_vec)+xf[0]
                        cone_y = r_vec[i]*np.sin(ang_vec)+xf[1]
                        plt.plot(cone_x, cone_y, cone_t_vec[i], '-c', alpha=0.15)

                    # Plot Settings
                    plt.xlim([map_size[0], map_size[1]])
                    plt.ylim([map_size[2], map_size[3]])
                    #plt.legend()
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('t [sec]')
                    plt.ioff() # Interactice Plot Trigger
                    plt.axis('equal')
                    plt.grid()
                    plt.show()  
                return G


    # Functios for Step 3
    def sphere_overlap(self, r, qnear, t_in):
        # Expanded equation to randomly select angle theta, which will compute the remainder of 3 by 1 vector for random point inbetween spheres
        # The random point is selected on the circular intersection of the sphere
        """
        th = rn.uniform(0, 2*np.pi)
        c1 = np.array([qnear[0], qnear[1], qnear[2]])
        c2 = np.array([self.xfin[0], self.xfin[1], self.xfin[2]])
        d = np.linalg.norm(c2-c1,2)
        r1 = r
        r2 = self.vmax*(self.endtime)
        print('r1', r1)
        print('r2', r2)
        h = 1/2 + (r1**2 - r2**2)/(2*d**2)

        print(r1+r2)
        print(d)
        print(bool(r1+r2 < d))

        print('h', h)
        print('d', d)
        print('hd', h*d)

        ci = c1 + h*(c2-c1)
        if r1**2 - (h*d)**2 >= 0:
            ri = np.sqrt(r1**2 - (h*d)**2)
        elif r1**2 - (h*d)**2 < 0:
            ri = np.sqrt((h*d)**2-r1**2)
        ni = (c2 - c1)/d
        
        ti = np.array([ci[0]+ri, ci[1], ci[2]])
        bi = np.cross(ni, ti)

        print('ti', ti)
        print('bi', bi)

        Pi = ci + ri*(ti*np.cos(th) + bi*np.sin(th))
        print('Pi', Pi)

        adfadfaf
        """
        
        # Choose random x, y within ri,
        while True:
            x = rn.uniform(1e-3, r)+qnear[0]
            y = rn.uniform(1e-3, r)+qnear[1]
            zsquared = r**2 - x**2 - y**2

            if zsquared >= 0:
                z = np.sqrt(zsquared)
                # Measure distance to two sphere's center, make sure the point is within radius
                r1 = r # Radius of Sphere from Qnear
                r2 = self.vmax*self.endtime # Radius of Sphere from Xf

                d1 = np.sqrt((x-qnear[0])**2 + (y-qnear[1])**2 + (z-qnear[2])**2)
                d2 = np.sqrt((x-self.xfin[0])**2 + (y-self.xfin[1])**2 + (z-self.xfin[2])**2)

                cond1 = bool(d1 <= r1)
                cond2 = bool(d2 <= r2)

                if cond1 and cond2:
                    break
        z = np.sqrt(zsquared)
        if z > self.xfin[2]:
            z = rn.uniform(1e-3, self.xfin[2])
        return x, y, z
            
    def gen_node(self, r, qnear, c, t_in):
        count = c
        # Generate qrand
        # Make sure sample is in intersection of two spheres
        qrand_x, qrand_y, qrand_z = self.sphere_overlap(r, qnear, t_in)

        # Obtain rotation angles
        qrand_phi = np.arctan2(self.xfin[2]-qnear[2], self.xfin[1]-qnear[1])
        qrand_th = np.arctan2(self.xfin[2]-qnear[2], self.xfin[0]-qnear[0])
        qrand_psi = np.arctan2(self.xfin[1]-qnear[1], self.xfin[0]-qnear[0])

        # Final Output
        #qrand = (qrand_x + qnear[0], qrand_y + qnear[1], qrand_z + qnear[2], qrand_phi, qrand_th, qrand_psi)
        qrand = (qrand_x, qrand_y, qrand_z, qrand_phi, qrand_th, qrand_psi)
        Qrand_SE3 = SE3(np.array([qrand[0], qrand[1], qrand[2], qrand[3], qrand[4], qrand[5]]))

        # check 2 things:
        # 1. qrand is within movement bound
        # 2. qrand is not in static map

        # With current setup, always assume max V,
        # So it will always in movement bound
        if c <= 100:
            if self.static_bound(qrand) and not Qrand_SE3 is None and not self.dynamic_bound(qrand, t_in):
                return Qrand_SE3
            else:
                count+=1
                return self.gen_node(r, qnear, count, t_in)
        else:
            # If loop is too much, just return stationary
            return SE3(qnear[0], qnear[1], qnear[2], qnear[3], qnear[4], qnear[5])
        
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
        # map size: [xmin, xmax, ymin, ymax, zmin, zmax]
        return (xi[0] >= self.map_st['size'][0] and
                xi[0] <= self.map_st['size'][1] and
                xi[1] >= self.map_st['size'][2] and
                xi[1] <= self.map_st['size'][3] and
                xi[2] >= self.map_st['size'][4] and
                xi[2] <= self.map_st['size'][5] )

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
            cam_i = cameras[str(i)]
            cam_tip_rad = 2*cam_i['Camera'].cam_radius()*np.sin(cam_i['Camera'].cam_FOV())
            cam_i = cam_i['FOV']
            cam_tip = cam_i[0,:]
            cam_dir = cam_i[0,:]-cam_i[1,:]
            h = np.linalg.norm(cam_dir, 2)
            r = cam_tip_rad
            p = np.array([xi[0], xi[1], xi[2]])
            cone_dist = np.dot(p-cam_tip, cam_dir)
            cone_radius = (cone_dist/h)*r
            orth_dist = np.linalg.norm((p-cam_tip) - cone_dist*cam_dir, 2)
            is_point_inside_cone = (orth_dist < cone_radius)
            check_vec.append(is_point_inside_cone)
        
        # Add all boolean values,
        # if 1: Point inside any camera FOVs
        # if 0: Point is outside of all camera FOVs
        return np.sum(check_vec)

    # Functions for Step 4
    def nearest(self, G, xi):
        # Find nearest vertex
        range_vec = []
        for ii in range(len(G['vertex'])):
            VERT_avail = G['vertex'][ii]
            vert_avail = SE3.to_vec(VERT_avail.to_matrix)
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
        diff = Xf.to_matrix - Q.to_matrix
        return np.sqrt((diff[0,2])**2 + (diff[1,2])**2 + (diff[2,2])**2)
    
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
    from Camera_3D import Camera
    from Dynamic_3D import DynamicMap

    # Test case
    # Initial, Final Positions
    x0 = np.array([-2.5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    x1 = np.array([12, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    t = [20, .1, 50]
    #t = [5, .1, 12.5] # Invalid tf Test

    # Vehicle Spec
    vehicle = {}
    vehicle['v'] = 1.25
    vehicle['radius'] = 0.5
    vehicle['w'] = np.deg2rad(30)

    # Map
    map_in = {}
    # Static Map
    map_in['st'] = {}
    map_in['st']['size'] = np.array([-5, 15, -15, 15, 0, 20])
    # Single buliding example
    map_in['st']['n'] = 1
    map_in['st']['0'] = np.array([
        (2.5,5,0), (7.5,5,0), (7.5,-5,0), (2.5,-5,0), (2.5,5,20), (7.5,5,20), (7.5,-5,20), (2.5,-5,20)
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
    cam_dict['z'] = cam_z
    cam_dict['a'] = np.deg2rad(45) # Tilt angle of camera facing downwards

    # Camera Spec
    tilt_limit = np.array([np.pi, 0]) #[upper, lower]
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
    RRT = test.R2T2_3D()
#%%