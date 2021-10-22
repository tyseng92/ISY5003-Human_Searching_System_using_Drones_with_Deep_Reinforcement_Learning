"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

import airsim
import time
import math
import cv2
import os
import numpy as np
import json
from inference_img import Yolov4

class DroneControl:
    def __init__(self, droneList):
        self.client = airsim.MultirotorClient('127.0.0.1')
        self.client.confirmConnection()
        self.droneList = droneList
        self.init_AirSim()
        self.image_dir = './captured_images/human_1/'
        self.target = 'human_1'
        self.snapshot_index = 0
        self.z_offset = self.get_spawn_z_offset(self.droneList[0])
        self.yolo = Yolov4()
    
    def init_AirSim(self):
        """
        Method to initialize AirSim for a list of drones
        """
        for drone in self.droneList:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
    
    def shutdown_AirSim(self):
        """
        Method to un-init all drones and quit AirSim
        """
        self.armDisarm(False)
        self.client.reset()
        self.enableApiControl(False)
    
    def resetAndRearm_Drones(self):
        """
        Method to reset all drones to original starting state and rearm
        """
        #self.armDisarm(False)
        self.client.reset()
        #self.enableApiControl(False)
        time.sleep(0.25)
        self.enableApiControl(True)
        self.armDisarm(True)

    def armDisarm(self, status):
        """
        Method to arm or disarm all drones
        """
        for drone in self.droneList:
            self.client.armDisarm(status, drone)
    
    def enableApiControl(self, status):
        """
        Method to enable or disable drones API control
        """
        for drone in self.droneList:
            self.client.enableApiControl(status, drone)

    def takeOff(self):
        """
        Method to take off for all drones
        """
        dronesClient = []
        for drone in self.droneList:
            cli_drone = self.client.takeoffAsync(vehicle_name=drone)
            dronesClient.append(cli_drone)
        for drone in dronesClient:
            drone.join()

    def getMultirotorState(self, drone):
        """
        Method to get current drone states
        """
        if drone in self.droneList:
            return self.client.getMultirotorState(vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getBarometerData(self, barometer, drone):
        """
        Method to get barometer data
        """
        if drone in self.droneList:
            return self.client.getBarometerData(barometer_name=barometer, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getImuData(self, imu, drone):
        """
        Method to get imu data
        """
        if drone in self.droneList:
            return self.client.getImuData(imu_name=imu, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getGpsData(self, drone):
        """
        Method to get gps data
        Returns GeoPoint object containing altitude, latitude and longitude
        """
        if drone in self.droneList:
            #return self.client.getGpsData(gps_name=gps, vehicle_name=drone)
            return self.client.getMultirotorState(vehicle_name=drone).gps_location
        else:
            print('Drone does not exists!')
    
    def getMagnetometerData(self, mag, drone):
        """
        Method to get Magnetometer data
        """
        if drone in self.droneList:
            return self.client.getMagnetometerData(magnetometer_name=mag, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getDistanceData(self, lidar, drone):
        """
        Method to get Distance data
        """
        if drone in self.droneList:
            return self.client.getDistanceSensorData(lidar_name=lidar, vehicle_name=drone)
        else:
            print('Drone does not exists!')

    def getLidarData(self, lidar, drone):
        """
        Method to get lidar data
        """
        if drone in self.droneList:
            return self.client.getLidarData(lidar_name=lidar, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getDronePos(self, drone):
        """
        Method to get X, Y, Z axis values of drone
        """
        if drone in self.droneList:
            x_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.x_val
            y_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.y_val
            z_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.z_val
            return np.array([x_val, y_val, z_val])
        else:
            print('Drone does not exists!')
    
    def moveDrone(self, drone, position, duration):
        """
        Method to move drone to indicated position
        pos = [x_val, y_val, z_val]
        """
        if drone in self.droneList:
            self.client.moveByVelocityAsync(vehicle_name=drone, 
                                             vx=position[0], 
                                             vy=position[1], 
                                             vz=position[2],
                                             duration=duration).join()
        else:
            print('Drone does not exists!')
            
    def simPause(self,pause):
        """
        Pass-through method to pause simulation
        """
        self.client.simPause(pause)
        
    def simGetCollisionInfo(self, drone):
        """
        Pass-through method to get collision info
        """
        return self.client.simGetCollisionInfo(drone)
    
    def hoverAsync(self, drone):
        """
        Pass-through method for hoverAsync
        """
        return self.client.hoverAsync(drone)

    def setCameraHeading(self, camera_heading, drone):
        """
        Set camera orientation
        """
        pos = self.getMultirotorState(drone).kinematics_estimated.position
        self.client.moveByVelocityZAsync(pos.x_val, pos.y_val, pos.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), vehicle_name=drone)
    
    def setCameraAngle_origin(self, camera_angle, drone, cam=0):
        """
        Set camera angle
        """
        pos = self.client.simSetCameraOrientation(cam, airsim.to_quaternion(
            camera_angle * math.pi / 180, 0, 0),vehicle_name=drone)  # radians
    
    def setCameraAngle(self, camera_angle, drone, cam="0"):
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(camera_angle), 0, 0)) #radians
        self.client.simSetCameraPose(cam, camera_pose, vehicle_name=drone)

    def getImage(self, drone, cam=0):
        """
        Get image for single drone
        """
        raw_img = self.client.simGetImage(cam, airsim.ImageType.Scene, vehicle_name=drone)
        return cv2.imdecode(airsim.string_to_uint8_array(raw_img), cv2.IMREAD_UNCHANGED)

    def captureImg(self, drone, cam = 0):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)  
        
        responses = self.client.simGetImages([airsim.ImageRequest(
            0, airsim.ImageType.Scene)],vehicle_name=drone)  # scene vision image in png format
        response = responses[0]
        filename = self.target + "_" + \
            str(self.snapshot_index) + "_" + str(int(time.time()))
        self.snapshot_index += 1
        airsim.write_file(os.path.normpath(
            self.image_dir + filename + '.png'), response.image_data_uint8)
        print("Saved snapshot: {}".format(filename))

    def inference(self, drone, cam = 0):
        responses = self.client.simGetImages([airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)],vehicle_name=drone)  # scene vision image in png format
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)
        results = self.yolo.predict(img_rgb)
        self.yolo.display()


    def turnDroneBySelfFrame(self, drone, turn_spd, duration):
        """
        Set camera orientation
        """
        pos = self.getMultirotorState(drone).kinematics_estimated.position
        self.client.moveByVelocityZBodyFrameAsync(0, 0, pos.z_val, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, turn_spd), vehicle_name=drone)

    def moveDroneBySelfFrame(self, drone, velocity, duration):
        """
        Method to move drone with indicated velocity
        velocity = [x_val, y_val, z_val]
        """
        if drone in self.droneList:
            self.client.moveByVelocityBodyFrameAsync(vehicle_name=drone, 
                                             vx=velocity[0], 
                                             vy=velocity[1], 
                                             vz=velocity[2],
                                             duration=duration).join()
        else:
            print('Drone does not exists!')

    # moveToPositionAsync(self, x, y, z, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(),
    #    lookahead = -1, adaptive_lookahead = 1, vehicle_name = '')
    def moveDroneToPos(self, drone, position):
        #z_offset = self.get_spawn_z_offset(drone)
        z = position[2]-self.z_offset
        self.client.moveToPositionAsync(vehicle_name=drone,
                                        x=position[0], y=position[1], z=z, velocity=1, timeout_sec=60, 
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                        yaw_mode=airsim.YawMode(True, 0)).join()

    def changeDroneAlt(self, drone, altitude):
        # getMultirotorState use the spawn coordinate rather than global coordinate from UE4, use offset to translate from UE4 to spawn coordinate (settings.json)
        #z_offset = self.get_spawn_z_offset(drone)
        pos = self.getMultirotorState(drone).kinematics_estimated.position
        print("init_alt:", pos.z_val)
        print("z_offset:", self.z_offset)
        z = altitude-self.z_offset
        print("z:",z)
        self.client.moveToPositionAsync(vehicle_name=drone,
                                        x=pos.x_val, y=pos.y_val, z=z, velocity=1, timeout_sec=60, 
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                        yaw_mode=airsim.YawMode(True, 0)).join()

    def check_pos_from_spawn(self, drone):
        pos = self.getMultirotorState(drone).kinematics_estimated.position
        print("spawn_position_info:", pos)
        return pos

    def check_pos_from_player_start(self, drone):
        #pos = self.client.simGetGroundTruthKinematics(vehicle_name=drone)
        pos = self.client.simGetObjectPose(drone)
        print("global_position_info:", pos)
        return pos

    def getSettingsString(self):
        string = self.client.getSettingsString()
        #print("Setting_json:", string)
        return string

    def get_spawn_z_offset(self, drone):
        s = self.getSettingsString()
        d = json.loads(s)
        z = d["Vehicles"][drone]["Z"]
        print("Z_offset:", z)
        return float(z)
                    