import numpy as np
import matplotlib.pyplot as plt
import random as rn
from Camera3D import Camera

class TBRRT:
    def __init__(self, cam_dict):
        self.x0 = 

        self.cam_dict = cam_dict
        pass

    def TBRRT3D(self):
        # Initial Variables
        x0 = self.x0
        xf = self.xf
        t_range = self.t_range

        repeat = True
        
        # Sensor Setup
        sensor_number = 1

        sensor_deployed = self.sensor_setup(sensor_number)


        # Vehicle Variables
        yawrate = np.deg2rad(30)
        pitchrate = np.deg2rad(30)
        velocity = 1

        # Storage
        Tree = {}
        Tree['x0'] = {}
        Tree['xf'] = {}
        Tree['x0']['Position'] = x0
        Tree['x0']['Time'] = 0
        Tree['x0']['Time Cost'] = 0
        Tree['x0']['Dist Cost'] = 0

        # Random Sample next node
        xprev = x0
        counter = 1
        while repeat:
            print('================================')
            print('Counter: '+str(counter))
            # Set Probability:
            sample = np.array([0.8, 0.2])
            
            # Vehicle Sample Domain
            rn_t = rn.uniform(0, t_range)

            # Randomly Sample x1
            rn_select = rn.uniform(0,1)
            nextNode = True
            while nextNode:
                if rn_select >= 0 and rn_select < sample[0]:
                    print('Random Sample')
                    rn_yawangle = rn.uniform(0, yawrate*rn_t)
                    rn_pitchangle = rn.uniform(0, pitchrate*rn_t)
                    R = velocity*rn_t
                    rn_x = xprev[0] + R*np.cos(rn_pitchangle)*np.cos(rn_yawangle)
                    rn_y = xprev[1] + R*np.cos(rn_pitchangle)*np.sin(rn_yawangle)
                    rn_z = xprev[2] + R*np.sin(rn_pitchangle)
                elif rn_select >= sample[0] and rn_select <= (sample[0] + sample[1]):
                    print('To Final')
                    normvec = (xf-x0)/np.linalg.norm(xf-x0, 2)
                    R = velocity*rn_t
                    travel = R*normvec
                    rn_x = xprev[0] + travel[0]
                    rn_y = xprev[1] + travel[1]
                    rn_z = xprev[2] + travel[2]

                # Check if vehicle's position is valid in parallel (return 1 when valid)
                # Check1: check if it is still in map
                # check2: check if it is in any building
                # check3: check if it is in FOV of sensors
                check_vec = np.zeros(1+building_number+sensor_number,)
                
                # Check 1
                if ((rn_x >= map_size[0] and rn_x <= map_size[1])&(rn_y >= map_size[2] and rn_y <= map_size[3])&(rn_z >= 0 and rn_z <= building_height)).all:
                    check_vec[0] = 1
                else:
                    check_vec[0] = 0

                # Check 2
                for nb in range (building_number):
                    building_i = building_storage[nb][0]
                    
                    # Repeat checking for building overlap
                    check_vec[1+nb] = bool(not (rn_x >= building_i[0] and rn_x <= building_i[1])&(rn_y >= building_i[2] and rn_y <= building_i[3]))

                # Check 3
                for nc in range(sensor_number):
                    conex = sensor[str(nc)].get_state()[0:3]
                    conedir = sensor[str(nc)].get_direction_vec(cam_dict['ang_init'][0], cam_dict['ang_init'][1])
                    p = np.array([rn_x, rn_y, rn_z])
                    cone_dist = np.dot(p-conex, conedir)
                    if cone_dist >= 0 and cone_dist <= sensor[str(nc)].get_h():
                        cone_radius = (cone_dist / sensor[str(nc)].get_h())*sensor[str(nc)].get_radius()
                        orth_dist = np.linalg.norm((p-conex)-cone_dist*conedir)
                        if orth_dist < cone_radius:
                            check_vec[1+building_number+nc] = 0
                        else:
                            check_vec[1+building_number+nc] = 1

                if sum(check_vec) == 1+building_number + sensor_number:
                    nextNode = False
                else:
                    nextNode = True
            print('Node: '+ str(rn_x) + ', ' + str(rn_y) + ', ' + str(rn_z))


            # Find closest node
            x1 = np.array([rn_x, rn_y, rn_z])
            dist_storage = []
            for i in range(counter):
                node_name_to_check = 'x'+str(i)
                dx = x1 - Tree[node_name_to_check]['Position']
                dist = np.sqrt((dx[0])**2 + (dx[1])**2 + (dx[2])**2)
                dist_storage.append(dist)
            min_dist_ind = np.where(dist_storage == min(dist_storage))
            prev_node_name = 'x'+str(min_dist_ind[0][0])

            # Log Child Node
            curr_node_name = 'x'+str(counter)
            Tree[prev_node_name]['Child'] = curr_node_name
            Tree[curr_node_name] = {}
            Tree[curr_node_name]['Parent'] = prev_node_name
            Tree[curr_node_name]['Position'] = x1
            Tree[curr_node_name]['Time'] = Tree[prev_node_name]['Time'] + rn_t
            Tree[curr_node_name]['Time Cost'] = Tree[prev_node_name]['Time Cost'] + rn_t
            Tree[curr_node_name]['Dist Cost'] = Tree[prev_node_name]['Dist Cost'] + R

            xprev = x1

            # Check for Destination

            # Update Counter
            counter += 1


    def sensor_setup(self, sensor_number):
        sensor = {}
        for nc in range(sensor_number):
            sensor[str(nc)] = Camera(self.cam_dict)
        return sensor