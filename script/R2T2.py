import numpy as np
import matplotlib.pyplot as plt
import random as rn
import shapely
from shapely import geometry
from mpl_toolkits import mplot3d

from SE2 import SE2, se2

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
        
        # t: 2x1 vector, [trange, delt]
        self.trange = t[0]
        self.delt = t[1]
        
        # vehcile: dictionary
        # key: v
        # 'v' = max velocity
        # 'radius' = radius
        self.vmax = vehicle['v']
        self.vrad = vehicle['radius']

        # map: dictionary
        # key: n, st, dy, curr
        # 'n' = repeat period length
        # 'st' = static map
        # ['st']['size'] = 4x1 vector [x0, xmax, y0, ymax]
        # ['st']['n'] = number of buildings
        # ['st'][str(i)], i = number // building poly  at ith index
        # 'dy' = repeating dynamic map with n periods
        # ['dy'][str(i)], i = number // dynamic map at ith index
        # 'curr' = static + dynamic = current map
        # ['curr'][str(i)], i = number = currnet map at ith index
        self.map_st = map_in['st']
        #self.map_dy = map_in['dy']
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
        map_size = self.map_st['size']
        bad_path = []

        # Start loop, and repeat until route toward final point
        repeat_R2T2 = True
        plot = True
        counter = 0

        # End condition
        dist_best = self.get_range(Xi, Xf)
        dist_tol = 1e-1

        if plot:
            fig = plt.figure(figsize=(5,5))
            
            for i in range(self.map_st['n']):
                ibuilding = self.map_st[str(i)]
                wall = geometry.LineString(ibuilding)
                building = geometry.Polygon(wall)
                bx,by = building.exterior.xy
                plt.plot(bx, by, '-k')

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
            switch = [.8, .1, .1]
            Qnear = None
            if Qnear is None:
                choose = rn.uniform(0,1)
                if choose >= 0 and choose <= switch[0]:
                    print('Random')
                    # Get random point x1, and check whether or not it is valid
                    Qnext = self.gen_node(mv_R, Qnear_prev.SE2param(), 0)
                elif choose > switch[0] and choose <= switch[2]:
                    print('Stationary')
                    # Statinoary
                    Qnext = Qnear_prev
                else:
                    print('Final')
                    if Qnear_prev in bad_path:
                        continue
                    else:
                        Qnext = SE2(xf[0], xf[1], xf[2])
            
            # Step 4
            # Find cloest point from qnear
            Qnear, tnear = self.nearest(G, Qnext.SE2param())

            print('Qnext ', Qnext.SE2param())
            print('Qnear ', Qnear.SE2param())
            print('trange ', trange)
            print('ti ', ti)

            # Step 5
            # Generate Path and check collision
            # This is done in SE2 Lie Group for 2D R2T2

            """
            Debug tool
            Qnear = SE2(0,0,np.pi/2)
            Qnext = SE2(2,3,np.pi/2)
            """

            Qroute = self.local_path_planner(Qnear, Qnext, self.vmax*ti)

            # Check Collision
            # If collision occur, we ignore that path
            # If collision doesn't occur, carry on
            if self.collision_check(self.vrad, Qnear, Qroute, G['t'], 21):
                bad_path.append(Qnear)
            else:
                # Compute current position
                # as time may differ
                Qcurr, path_curr = self.current_pos(Qnear, Qroute)

                if plot:
                    # Plot Qnext
                    qnpt = Qnext.SE2param()
                    #plt.plot(qnpt[0], qnpt[1], '.r', alpha = .5)

                    # Plot Qcurr
                    qcpt = Qcurr.SE2param()
                    plt.plot(qcpt[0], qcpt[1], '+r')

                    # Plot Route
                    V = Qnear.inv()@(Qroute)
                    v = SE2_log(V)
                    p_path = []

                    for tt in np.linspace(0,1,11):
                        Q = Qnear.matrix()@(SE2_exp(v*tt))
                        xt, yt, thetat = SE2_to_param(Q)
                        p_path.append([xt, yt])
                        if np.sqrt((qnpt[0]-xt)**2 + (qnpt[1]-yt)**2) <= 1e-2:
                            break
                        
                    p_path = np.array(p_path)
                    plt.plot(p_path[:,0], p_path[:,1], '-r', linewidth=5, alpha=.25)

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

            # If within tolerance, we stop and return path
                if dist_best < dist_tol:
                    print('Destination Reached')
                    # Stop iteration
                    repeat_R2T2 = False

                    # Find route to Final Point
                    route = self.rrt_course(G)
                    G['route'] = route

                    if plot:
                        # Plot route
                        cawl = G['edge']
                        for ri in route:
                            ed = cawl[str(ri)]
                            plt.plot(ed['x'], ed['y'], '--r')

                        # Plot xi, xf
                        plt.plot(xi[0], xi[1], 'xr')
                        plt.plot(xf[0], xf[1], 'xb')

                        # Plot Settings
                        plt.xlim([map_size[0], map_size[1]])
                        plt.ylim([map_size[2], map_size[3]])
                        plt.show()
                    
                    return G
            # If not, we update for next loop
                print('\n')
                counter += 1

                # Break out if loop is too long
                if counter > 50:
                    print('Fail to reach destination in time')

                    if plot:
                        # Plot xi, xf
                        plt.plot(xi[0], xi[1], 'xr')
                        plt.plot(xf[0], xf[1], 'xb')

                        # Plot Settings
                        plt.xlim([map_size[0], map_size[1]])
                        plt.ylim([map_size[2], map_size[3]])
                        plt.show()
                    return G


    # Functios for Step 3
    def gen_node(self, r, qnear, c):
        count = c
        # Generate qrand
        qrand_x = rn.uniform(-r, r)
        qrand_y = np.sqrt(r**2 - qrand_x**2)
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
            if self.static_bound(qrand) and not Qrand_SE2 is None:
                return Qrand_SE2
            else:
                count+=1
                return self.gen_node(r,qnear, count)
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
        return (xi[0] >= self.map_st['size'][0] and xi[0]<self.map_st['size'][1]+1 and xi[1] >= self.map_st['size'][2] and xi[1]<self.map_st['size'][3]+1)

    def static_bound(self, xi):
        return self.in_map(xi) and not self.building_bound(xi)
    
    # Functions for Step 4
    def nearest(self, G, xi):
        # Find nearest vertex
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
        M = (X0.inv()).dot(X1.matrix())
        dth = np.arctan2(M[1,0], M[0,0])
        dx = M[0,2]
        dy = M[1,2]

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
        return X0.matrix().dot(V)
    
    def current_pos(self, Q0, Q1):
        V = Q0.inv()@(Q1)
        v = SE2_log(V)
        curr_path = {}
        curr_path['x'] = []
        curr_path['y'] = []
        curr_path['theta'] = []
        for tt in np.linspace(0,1,11):
            Q = Q0.matrix()@(SE2_exp(v*tt))
            xt, yt, thetat = SE2_to_param(Q)
            curr_path['x'].append(xt)
            curr_path['y'].append(yt)
            curr_path['theta'].append(thetat)
        return SE2(x=xt, y=yt, theta=thetat), curr_path
    
    def collision_check(self, vehicle_radius, Q0, Q1, qtvec, steps):
        V = Q0.inv()@(Q1)
        v = SE2_log(V)
        for tt in np.linspace(0,1,steps):
            Q = Q0.matrix()@(SE2_exp(v*tt))
            xt, yt, thetat = SE2_to_param(Q)
            
            # check map bounds
            if not self.in_map([xt, yt, thetat]):
                return True

            # check collision with static obstacles
            if self.in_map([xt, yt, thetat]):                
                if bool(self.building_bound([xt, yt, thetat])):
                    return True
            # check collision with dynamic obstacles
            # at given time vector qtvec
        return False
    
    # Functions for Step 6
    def get_range(self, Q, Xf):
        diff = Xf.matrix() - Q.matrix()
        return np.sqrt((diff[0,2])**2 + (diff[1,2])**2)
    
    # Functions for Step 7
    def rrt_course(self, G):
        neighbor = G['neighbor']

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
            for ii in range(len(neighbor)):
                if match_vertex(qcheck, neighbor[str(ii)][1]):
                    route.append(ii)
                    qcheck = neighbor[str(ii)][0]
                    # Return route to final point
                    if match_vertex(qcheck, q0):
                        castellan = False
                        return route

#%%
if __name__ == "__main__":
    import numpy as np

    # Test case
    # Initial, Final Positions
    x0 = np.array([0, 0, np.pi/2])
    x1 = np.array([10, 0, np.pi/2])
    t = [10, 1]

    # Vehicle Spec
    vehicle = {}
    vehicle['v'] = 1
    vehicle['radius'] = 0.5

    # Map
    map_in = {}
    # Static Map
    map_in['st'] = {}
    map_in['st']['size'] = np.array([-5, 15, -10, 10])
    # Single buliding example
    map_in['st']['n'] = 1
    map_in['st']['0'] = np.array([
        (2.5,5), (7.5,5), (7.5,-5), (2.5,-5)
    ])

    # Dynamic Map
    map_in['n'] = 10
    map_in['dy'] = {}

    # Camera Position
    # TODO
    # Setup Camera and make continuous dynamic map

    # Camera Spec

    dynamic_map_time_vec = np.arange(0, t[0], t[1])
    for ti in dynamic_map_time_vec:
        map_in['dy'][str(ti)]


    test = R2T2(x0, x1, t, vehicle, map_in)
    RRT = test.R2T2_2D()
#%%