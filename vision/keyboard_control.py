import airsim
import os
import time
from pynput import keyboard
from DroneControlAPI import DroneControl
from absl import app, flags, logging
from absl.flags import FLAGS

class MoveDrone(object):
    def __init__(self, drone_list):
        self.dc = DroneControl(drone_list)
        self.dc.takeOff()
        self.target_drone = drone_list[0]
        self.spd = 1
        self.angle_spd = 10
        self.move_time = 0.1
        self.camera_angle = -50
        # remember change altitude to negative as airsim use NED frame, where negative means upward.
        self.altitude = -2.5
        self.init_pos = [0,0,self.altitude]
        # check position
        self.dc.check_pos_from_player_start(self.target_drone)

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
                '4':self.change_alt_top,
                '5':self.change_alt_bottom,
                'x':self.check_position
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
        self.camera_angle += 5
        self.dc.setCameraAngle(self.camera_angle, self.target_drone, cam="0")
        print("camera_angle:", self.camera_angle)

    def cam_down(self):
        self.camera_angle -= 5
        self.dc.setCameraAngle(self.camera_angle, self.target_drone, cam="0")
        print("camera_angle:", self.camera_angle)

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
        self.dc.captureImg(self.target_drone)
    
    def inference(self):
        self.stop()
        self.dc.inference(self.target_drone)

    def stop(self):
        #print(self.target_drone)
        self.dc.moveDroneBySelfFrame(self.target_drone, [0,0,0], self.move_time)
        self.dc.hoverAsync(self.target_drone)
        self.stabilize()

def main(_argv):
    droneList = ['ShooterDrone']
    md = MoveDrone(droneList)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    