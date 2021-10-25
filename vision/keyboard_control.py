import airsim
import os
import time
from pynput import keyboard
from DroneControlAPI import DroneControl
from absl import app, flags, logging
from absl.flags import FLAGS

class MoveDrone(object):
    def __init__(self, drone_list, drone_id=0, inference=True):
        self.dc = DroneControl(drone_list, drone_id, inference)
        self.inference = inference
        self.dc.takeOff()
        self.target_drone = drone_list[drone_id]
        self.spd = 1
        self.angle_spd = 10
        self.move_time = 0.1
        # [[front_right cam orientation], [front_left cam orientation] ,[back_center cam orientation]]
        self.camera_angle = [[-50, 0, 60], [-50, 0, -60], [-50, 0, 0]] 
        # remember change altitude to negative as airsim use NED frame, where negative means upward.
        self.altitude = -2.5
        self.init_pos = [0,0,self.altitude]
        # check position
        self.dc.check_pos_from_player_start(self.target_drone)

        # change cams angle
        self.dc.setCameraAngle(self.camera_angle[0], self.target_drone, cam="1")
        self.dc.setCameraAngle(self.camera_angle[1], self.target_drone, cam="2")
        self.dc.setCameraAngle(self.camera_angle[2], self.target_drone, cam="4")

        # initialize position
        self.dc.moveDroneToPos(self.target_drone, self.init_pos)
        self.dc.check_pos_from_player_start(self.target_drone)
        # k_init must be at the last line of __init__()
        self.kb_init()

    # keyboard functions
    def kb_init(self):
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()
        self.dc.shutdown_AirSim()

    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            self.run(key.char)
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    def on_release(self, key):
        print('{0} released'.format(
            key))
        self.stop()
        # press esc to end
        if key == keyboard.Key.esc:
            return False

    def run(self, char):
        switcher={
                'w':self.front,
                's':self.back,
                'z':self.left,
                'c':self.right,
                'a':self.clockwise,
                'd':self.anticlockwise,
                'q':self.top,
                'e':self.bottom,
                'f':self.inference,
                'g':self.capture,
                '1':self.cam_up,
                '3':self.cam_down,
                '9':self.cam_left,
                '0':self.cam_right,
                '4':self.change_alt_top,
                '5':self.change_alt_bottom,
                'x':self.check_position,
                't':self.gps_check
                }
        func=switcher.get(char,lambda :'Invalid Key!')
        return func()

    def front(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [self.spd,0,0], self.move_time)
        #self.stabilize()

    def back(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [-self.spd,0,0], self.move_time)
        #self.stabilize()

    def left(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,-self.spd,0], self.move_time)

    def right(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,self.spd,0], self.move_time)

    def top(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,0,self.spd], self.move_time)

    def bottom(self):
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,0,-self.spd], self.move_time)

    def clockwise(self):
        self.dc.turnDroneBySelfFrame(self.target_drone, -self.angle_spd, self.move_time)

    def anticlockwise(self):
        self.dc.turnDroneBySelfFrame(self.target_drone, self.angle_spd, self.move_time)

    def cam_up(self):
        self.camera_angle[0][0] += 5
        self.camera_angle[1][0] += 5
        self.camera_angle[2][0] += 5
        self.dc.setCameraAngle(self.camera_angle[0], self.target_drone, cam="1")
        self.dc.setCameraAngle(self.camera_angle[1], self.target_drone, cam="2")
        self.dc.setCameraAngle(self.camera_angle[2], self.target_drone, cam="4")
        print("camera_angle:", self.camera_angle[0][0])

    def cam_down(self):
        self.camera_angle[0][0] -= 5
        self.camera_angle[1][0] -= 5
        self.camera_angle[2][0] -= 5
        self.dc.setCameraAngle(self.camera_angle[0], self.target_drone, cam="1")
        self.dc.setCameraAngle(self.camera_angle[1], self.target_drone, cam="2")
        self.dc.setCameraAngle(self.camera_angle[2], self.target_drone, cam="4")
        print("camera_angle:", self.camera_angle[0][0])

    def cam_left(self):
        self.camera_angle[0][2] -=5
        self.camera_angle[1][2] -= 5
        self.camera_angle[2][2] -= 5
        self.dc.setCameraAngle(self.camera_angle[0], self.target_drone, cam="1")
        self.dc.setCameraAngle(self.camera_angle[1], self.target_drone, cam="2")
        self.dc.setCameraAngle(self.camera_angle[2], self.target_drone, cam="4")
        print("camera_angle:", self.camera_angle[0][2])

    def cam_right(self):
        self.camera_angle[0][2] += 5
        self.camera_angle[1][2] += 5
        self.camera_angle[2][2] += 5
        self.dc.setCameraAngle(self.camera_angle[0], self.target_drone, cam="1")
        self.dc.setCameraAngle(self.camera_angle[1], self.target_drone, cam="2")
        self.dc.setCameraAngle(self.camera_angle[2], self.target_drone, cam="4")
        print("camera_angle:", self.camera_angle[0][2])

    def change_alt_top(self):
        # remember change altitude to negative as airsim use NED frame, where negative means upward.
        self.altitude -= 0.5
        self.dc.changeDroneAlt(self.target_drone, self.altitude)
        print("altitude:", self.altitude)

    def change_alt_bottom(self):
        # remember change altitude to negative as airsim use NED frame, where negative means upward.
        self.altitude += 0.5
        self.dc.changeDroneAlt(self.target_drone, self.altitude)
        print("altitude:", self.altitude)

    def check_position(self):
        # AirSim:        distance unit => m 
        #                altitude => upward is negative
        # Unreal Engine: distance unit => cm
        #                altitude => upward is positive
        self.dc.check_pos_from_player_start(self.target_drone)

    def stabilize(self):
        #print("stabilize")
        time.sleep(0.1)
        self.top()
        time.sleep(0.1)
        self.bottom()

    def capture(self):
        self.stop()
        # change cam id to 1, 2, or 4
        self.dc.captureImg(self.target_drone, cam = 1)
    
    def inference(self):
        if self.inference:
            self.stop()
            # change cam id to 1, 2, or 4
            self.dc.inference(self.target_drone, cam = 1)

    def stop(self):
        #print(self.target_drone)
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,0,0], self.move_time)
        self.dc.hoverAsync(self.target_drone)
        self.stabilize()

    def gps_check(self):
        gps = self.dc.getGpsData(self.target_drone)
        print("gps:", gps)

def main(_argv):
    droneList = ['Drone0', 'Drone1', 'Drone2']
    #md = MoveDrone(droneList, drone_id=0)
    # change drone with drone_id: 0, 1, or 2, set inference to 'True' to use inference function
    md = MoveDrone(droneList, drone_id=0, inference=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    