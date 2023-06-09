import numpy as np
import matplotlib.pyplot as plt
import random as rn
import shapely
import json
from shapely import geometry

# 3D Plot
from mpl_toolkits import mplot3d

# Custom Classes
from SE2_Legacy import SE2, se2

#%%
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import random as rn
    import shapely
    import json
    from shapely import geometry

    # 3D Plot
    from mpl_toolkits import mplot3d

    # Custom Classes
    from SE2_Legacy import SE2, se2
    from Camera import Camera
    from Dynamic import DynamicMap

    """# Setup Map
    x0 = np.array([0, 0, np.pi/2])
    x1 = np.array([12.5, 0, -np.pi/2])
    t = [20, .1]

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
    map_in['st']['0'] = np.array([
        (2.5,5), (7.5,5), (7.5,-15), (2.5,-15)
    ])

    # Dynamic Map
    # This is a continuous function generates camera FOV coverages
    # Input is map_in, and time input t_in
    map_in['n'] = 100#t[0]
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
    fov_ang = np.deg2rad(35)
    fov_rng = 7.5 #[m]
    cam_period = t[0]
    cam_increment = t[1]
    cam_dict['spec'] = {}
    cam_dict['spec']['bound'] = tilt_limit
    cam_dict['spec']['fov'] = [fov_ang, fov_rng]
    cam_dict['spec']['cam_time'] = [cam_period, cam_increment]"""

    # Test case
    # Initial, Final Positions
    x0 = np.array([10, -10, -np.pi])
    x1 = np.array([10, 10, 0])
    t = [30, .1]

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
    map_in['st']['n'] = 2
    #map_in['st']['0'] = np.array([(2.5,5), (7.5,5), (7.5,-5), (2.5,-5)])
    map_in['st']['0'] = np.array([
        (-5,15), (-2.5,15), (-2.5,-15), (-5,-15)
    ])
    map_in['st']['1'] = np.array([
        (2.5,-1.5), (15, -1.5), (15,1.5), (2.5,1.5)
    ])

    # Dynamic Map
    # This is a continuous function generates camera FOV coverages
    # Input is map_in, and time input t_in
    map_in['n'] = t[0]
    map_in['ncam'] = 1

    # Single camera example, surveying final location xfin
    # Camera Position
    cam_x = np.array([-2.5])
    cam_y = np.array([0])
    cam_dict = {}
    cam_dict['n'] = len(cam_x)
    cam_dict['x'] = cam_x
    cam_dict['y'] = cam_y

    # Camera Spec
    tilt_limit = np.array([np.pi, 0]) #[upper, lower]
    fov_ang = np.deg2rad(10)
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


    # Import data
    with open('RRT_Result.json') as json_file:
        RRT = json.load(json_file)

    with open('R2T2_Result_Wait_Go.json') as json_file:
        R2T2 = json.load(json_file)

    

    # Visualization
    fig = plt.figure(figsize=(7.5,7.5))
    #ax = plt.axes(projection ='3d')
    ax = fig.add_subplot(111)#,projection='3d')
    
    # Graph building
    for i in range(map_in['st']['n']):
        ibuilding = map_in['st'][str(i)]
        wall = geometry.LineString(ibuilding)
        building = geometry.Polygon(wall)
        bx,by = building.exterior.xy
        plt.plot(bx, by, '-k', label = 'Building')

    # Graph camera at t0
    cameras_at_t0 = dmap.gen_cam(R2T2['t'][0])
    for i in range(cameras_at_t0['n']):
        camera_i_FOV = cameras_at_t0[str(i)]['FOV']
        camera_i_FOV_Poly = cameras_at_t0[str(i)]['FOV_Poly']
        # Plot Camera Location
        plt.plot(camera_i_FOV[0,0], camera_i_FOV[0,1], 'og', label = 'Camera')
        # Plot FOV Polygon
        FOV_Poly_xi_t0, FOV_Poly_yi_t0 = camera_i_FOV_Poly.exterior.xy
        plt.plot(FOV_Poly_xi_t0, FOV_Poly_yi_t0, '-g')
    
    # Plot Static Map
    """R2T2_tvec = np.arange(0, R2T2['t'][-1], 0.1)
    for i in range(map_in['st']['n']):
        ibuilding = map_in['st'][str(i)]
        wall = geometry.LineString(ibuilding)
        building = geometry.Polygon(wall)
        bx,by = building.exterior.xy
        for st_i in R2T2_tvec:
            # Currently set alpha = 0
            plt.plot(bx, by, st_i, '-k', alpha=0.1)
    for i in range(5):
        plt.plot([bx[i], bx[i]], [by[i], by[i]], [0, st_i], '-k')
    plt.plot(bx, by, st_i, '-k', alpha=1)

    # Plot Dynamic Map
    for st_i in R2T2_tvec:
        cameras = dmap.gen_cam(st_i)
        ncam = cameras['n']
        for ii in range(ncam):
            cam_i = cameras[str(ii)]['FOV_Poly']
            cx, cy = cam_i.exterior.xy
            plt.plot(cx, cy, st_i, '-g', alpha=0.25)
    plt.plot(cx, cy, st_i, '-g', alpha=1)"""

    """## RRT
    # Plot Point
    # 3D Plot
    for ii in range(2):
        # 3D
        # Plot Qnear
        qrpt = RRT['vertex'][ii]
        #plt.plot(qrpt[0], qrpt[1], RRT['t'][ii], '.r', alpha=.5)

        # Plot Qnext
        qnpt = RRT['vertex'][ii+1]
        #plt.plot(qnpt[0], qnpt[1], RRT['t'][ii+1], 'xr', alpha = .5)

        # Plot Qnext
        #qcpt = Qcurr.SE2param()
        #plt.plot(qcpt[0], qcpt[1], tnear+tt, 'xr', alpha = .5, label='Next vertex', zorder=15)

        # Plot Current Path
        path_curr = RRT['edge'][str(ii)]
        #plt.plot(path_curr['x'], path_curr['y'], path_curr['t'],'-r', linewidth=5, alpha=.5)

        # 2D
        # Plot Qnear
        plt.plot(qrpt[0], qrpt[1], '.r', alpha=.5)

        # Plot Qnext
        plt.plot(qnpt[0], qnpt[1], 'xr', alpha = .5)
        
        # Plot Current Path
        plt.plot(path_curr['x'], path_curr['y'], '-r', linewidth=5, alpha=.5)
    #plt.plot(path_curr['x'], path_curr['y'], path_curr['t'],'-r', linewidth=5, alpha=.5, label='3D Path for RRT')
    plt.plot(path_curr['x'], path_curr['y'], '-r', linewidth=5, alpha=.5, label='2D Path for RRT')

    ## RRT
    # Plot Point
    # 3D Plot
    for i in range(len(R2T2['vertex'])-1):
        # 3D
        # Plot Qnear
        if R2T2['t'][i+1] <= R2T2['t'][-1]:
            #qrpt = R2T2['vertex'][i]
            #plt.plot(qrpt[0], qrpt[1], R2T2['t'][i], '.b', alpha=.5, label='Nearest vertex', zorder=15)

            # Plot Qnext
            qnpt = R2T2['vertex'][i+1]
            #plt.plot(qnpt[0], qnpt[1], R2T2['t'][i+1], 'xb', alpha = .5)
            plt.plot(qnpt[0], qnpt[1], 'xb', alpha = .5)"""

        # Plot Current Path
    #for key in list(R2T2['edge'].keys()):
    #    path_curr = R2T2['edge'][key]
    #    if path_curr['t'][-1] <= R2T2['t'][-1]:
    #        plt.plot(path_curr['x'], path_curr['y'], path_curr['t'],'-b', linewidth=1, alpha=.15, label='Computed route', zorder=15)
    #        # Plot Current Path
    #        plt.plot(path_curr['x'], path_curr['y'], '-c', linewidth=1, alpha=.5, label='Comptued route in 2D', zorder=15)

    # Plot Route for R2T2
    route_R2T2 = R2T2['route']
    route_tvec = []
    for ri in np.flip(route_R2T2):
        route_tvec.append(R2T2['t'][ri])
    route_tvec.append(route_R2T2)

    cawl = R2T2['edge']
    cawl_key = list(cawl.keys())
    cawl_counter = 0
    for ri in np.flip(route_R2T2[0:len(route_R2T2)]):
        ed = cawl[str(cawl_key[ri])]
        #plt.plot(ed['x'], ed['y'], ed['t'], '--b', zorder=10)
        #plt.plot(ed['x'], ed['y'], ed['t'], '-b', linewidth=5, alpha=.5)
        #plt.plot(ed['x'], ed['y'], '--c', zorder=10)
        plt.plot(ed['x'], ed['y'], '-b', linewidth=5, alpha=.5)
        cawl_counter += 1
        if cawl_counter == len(route_R2T2)-1:
            break
    #plt.plot(ed['x'], ed['y'], ed['t'], '-b', linewidth=5, alpha=.5, label='3D Path for R2T2')
    plt.plot(ed['x'], ed['y'], '-b', linewidth=5, alpha=0, label='2D Path for R2T2')

    

    # Plot xi, xf
    plt.plot(x0[0], x0[1], 'xr')
    plt.plot(ed['x'][-1], ed['y'][-1], 'xb')
    #plt.plot(x0[0], x0[1], 0, '^r')
    #plt.plot(x1[0], x1[1], R2T2['t'][-1], '^b')
    
    #plt.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    #ax.set_zlabel('t [sec]')
    #ax.set_zlim(0, R2T2['t'][-1])
    plt.xlim([map_in['st']['size'][0], map_in['st']['size'][1]])
    plt.ylim([map_in['st']['size'][2], map_in['st']['size'][3]])
    plt.legend()
    plt.ioff()
    plt.show()
# %%
