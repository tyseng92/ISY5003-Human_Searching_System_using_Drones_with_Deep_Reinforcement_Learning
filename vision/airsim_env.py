import time
import numpy as np
import airsim
import config
from geopy import distance
from DroneControlAPI import DroneControl
from keyboard_control import MoveDrone
from inference_img import Yolov4

import math
from pathlib import Path

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5

# base on UE4 coordinate with NED frame
floorZ = 0 
min_height = -3 
max_height = -8

goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

droneList = ['Drone0', 'Drone1', 'Drone2']
#base_dir = Path('..')
#yolo_weights = base_dir/'weights'/'drone.h5'

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.dc = DroneControl(droneList)
        # Load the inference model
        #self.infer_model = YoloPredictor(yolo_weights)
        self.yolo = Yolov4()
        self.action_size = 3
        self.altitude = -2.5
        self.init_pos = [0,0,self.altitude]
        self.camera_angle = [-50, 0, 0]
        self.level = 0
        #self.responses = []
        #self.drone_pos = []

    def reset(self):
        '''
        Method to reset AirSim env to starting position
        '''
        self.level = 0
        self.dc.resetAndRearm_Drones()

        # all drones takeoff
        self.dc.simPause(False)
        for drone in droneList:
            print(f'{drone} taking off...')
            #self.dc.moveDrone(drone, [0,0,-1], 2 * timeslice)
            #self.dc.moveDrone(drone, [0,0,0], 0.1 * timeslice)
            self.dc.moveDroneToPos(drone, self.init_pos)
            self.dc.hoverAsync(drone).join()
            self.dc.setCameraAngle(self.camera_angle, drone, cam="0")

        # capture image to numpy by drone
        responses = []
        for drone in droneList:
            img = self.dc.captureImgNumpy(drone, cam = 0)
            responses.append(img)

        drone_pos = []
        for drone in droneList:
            drone_pos.append(self.dc.getDronePosition(drone)) 

        observation = [responses, drone_pos]
        return observation

    def step(self, quad_offset_list):
        # move with given velocity
        quad_offset = []
        for qoffset in quad_offset_list: # [(xyz),(xyz),(xyz)]
            quad_offset.append([float(i) for i in qoffset])
        self.dc.simPause(False)
        
        # Move the drones
        for id, drone in enumerate(droneList):
            self.dc.moveDrone(drone, [quad_offset[id][0], quad_offset[id][1], quad_offset[id][2]], 2* timeslice)

        # Get follower drones position and linear velocity        
        landed = [False, False, False]
        collided = [False, False, False]
        has_collided = [False, False, False]
        collision_count = [0, 0, 0]

        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = []
            quad_vel = []
            for drone in droneList:
                #quad_pos.append(self.dc.getMultirotorState(drone).kinematics_estimated.position)
                quad_pos.append(self.dc.getDronePosition(drone))
                quad_vel.append(self.dc.getMultirotorState(drone).kinematics_estimated.linear_velocity)

            # decide whether collision occured
            for id, drone in enumerate(droneList):
                collided[id] = self.dc.simGetCollisionInfo(drone).has_collided
                land = (quad_vel[id].x_val == 0 and quad_vel[id].y_val == 0 and quad_vel[id].z_val == 0)
                landed[id] = land or quad_pos[id].z_val > floorZ
                collision = collided[id] or landed[id]
                if collision:
                    collision_count[id] += 1
                if collision_count[id] > 10:
                    has_collided[id] = True
                    break
            if any(has_collided):
                break

        self.dc.simPause(True)
        #time.sleep(1)

        # All of the drones take image
        responses = []
        for drone in droneList:
            img = self.dc.captureImgNumpy(drone, cam = 0)
            responses.append(img)

        drone_pos = []
        for drone in droneList:
            drone_pos.append(self.dc.getDronePosition(drone)) 
        
        # Get each follower drone image reward
        exist_reward = {}
        focus_reward = {}
        for id, drone in enumerate(droneList):
            img = responses[id]
            #try:
            bbox = self.dc.getPredBbox(img)
            # if no detection is found, where bbox is [0,0,0,0].
            if not any(bbox):
                exist_status = 'miss'
                exist_reward[id] = exist_status
            # if there is detection found in image.
            else:
                exist_status = 'found'
                exist_reward[id] = exist_status

                focus_status = self.check_focus(bbox, img)
                focus_reward[id] = focus_status

                size_status = self.check_size(bbox)
                size_reward[id] = size_status

            print(f'Drone[{id}] status: [{exist_status}], [{focus_status}], [{size_status}]')

        # decide if episode should be terminated
        done = False
        # fly below min height or above max height
        out_range = [False, False, False]
        for id, drone in enumerate(droneList):
            if drone_pos[id].z_val > min_height or drone_pos[id].z_val < max_height: 
                out_range[id] = True

        done = any(has_collided) or any(out_range)     

        # compute reward
        reward = self.compute_reward(responses, exist_reward, focus_reward, size_reward, done)

        # log info
        loginfo = []
        for id, drone in enumerate(droneList):
            info = {}
            info['Z'] = drone_pos[id].z_val
            info['level'] = self.level
            if landed[id]:
                info['status'] = 'landed'
            elif has_collided[id]:
                info['status'] = 'collision'
            elif exist_reward[id] == 'miss':
                info['status'] = 'miss'
            elif exist_reward[id] == 'found':
                info['status'] = 'found'    
            elif any(out_range):
                info['status'] = 'dead'
            else:
                info['status'] = 'none'
            loginfo.append(info)
            observation = [responses, drone_pos]

        return observation, reward, done, loginfo

    def check_focus(self, bbox, image):
        image_h, image_w, _ = image.shape
        box = bbox[0][0][0] 
        # from ratio to pixel
        box[0] = int(box[0] * image_h)
        box[2] = int(box[2] * image_h)
        box[1] = int(box[1] * image_w)
        box[3] = int(box[3] * image_w)

        # tensorflow uses [ymin, xmin, ymax, xmax] convention for bounding box
        ymin, xmin, ymax, xmax = box
        c_x = xmin + ((xmax - xmin) / 2)
        c_y = ymin + ((ymax - ymin) / 2)

        img_cen_x = image_w / 2
        img_cen_y = image_h / 2

        # Check if the center of the detection box is within the bounding box 'fbbox' 
        fbbox = {
            'xmin': img_cen_x - (image_w * 0.2 / 2), # Xmin
            'xmax': img_cen_x + (image_w * 0.2 / 2), # Xmax
            'ymin': img_cen_y - (image_h * 0.2 / 2), # Ymin
            'ymax': img_cen_y + (image_h * 0.2 / 2)  # Ymax
        }

        if (c_x > fbbox['xmin'] and c_x < ffbox['xmax']) and (c_y > fbbox['ymin'] and c_y < ffbox['ymax']):
            status = 'in'
        else:
            status = 'out'            
        return status

    def check_size(self, bbox):
        box = bbox[0][0][0]
        y_delta = box[2] - box[0]
        x_delta = box[3] - box[1]
        # if the bbox size is larger than certain size, then it will be in 'large' status, otherwise in 'small' status.
        if y_delta > 0.8 and x_delta > 0.25:
            status = 'large'
        else:
            status = 'small'
        return status

    def compute_reward(self, responses, exist_reward, focus_reward, size_reward, done):
        reward = [None] * len(droneList)
        for id, drone in enumerate(droneList):         
            img = responses[id]
            exist_status = exist_reward[id]
            focus_status = focus_reward[id]
            size_status = size_reward[id]
            
            # Assign reward value based on status
            if done:
                reward[id] = config.reward['dead']
            elif exist_status == 'miss':
                reward[id] = config.reward['miss']
            elif exist_status == 'found' and focus_status == 'in' and size_status == 'large':
                reward[id] = config.reward['large_in']
            elif exist_status == 'found' and focus_status == 'out' and size_status == 'large':
                reward[id] = config.reward['large_out']
            elif exist_status == 'found' and focus_status == 'in' and size_status == 'small':
                reward[id] = config.reward['small_in']
            elif exist_status == 'found' and focus_status == 'out' and size_status == 'small':
                reward[id] = config.reward['small_out']
            else:
                reward[id] = config.reward['none']

            # Append GPS rewards
            if img_status != 'dead':            
                gps = gps_dist[droneidx]
                if gps > 9 or gps < 2.3:
                    reward[id] = reward[id] + config.reward['dead']
                else:
                    reward[id] = reward[id] + config.reward['forward']
        return reward
    
    
    def disconnect(self):
        self.dc.shutdown_AirSim()
        print('Disconnected.')