import config

def drone_reward_total_coverage()

    gps_drone1 = self.dc.getGpsData(droneList[0])
    gps_drone2 = self.dc.getGpsData(droneList[1])
    gps_drone3 = self.dc.getGpsData(droneList[2])
    gps_drone4 = self.dc.getGpsData(droneList[3])
    # gps_droneTarget = self.dc.getGpsData(droneList[3])

    # repurposing to calculation of  area for reward function instead of comparison of distance between different drones

    gps_dist = []
    dist_covered = []
    responses = []
    target = (gps_droneTarget.latitude, gps_droneTarget.longitude)
    for x in gps_dist:
        source = (x.latitude, x.longitude)
        gps_dist.append(distance.distance(source,  # starting_point).m)
    for x in gps_dist:
        dist_covered.append(x * camera_coverage_of_area)
    total_distance = sum(gps_dist)
    for x in gps_dist:
        if x < (total_distance/3):
            responses.append("slow")

    observation = [gps_dist,dist_covered,responses]
    return observation



def drone_calculate_rewards(observation):
        reward = [None] * len(droneList[:-1])
        for droneidx in range(len(droneList[:-1])):


            # Assign reward value based on status
            # if dead or img_status == 'dead':
            #     reward[droneidx] = config.reward['dead']
            if slow or responses="slow"
                  reward[droneidx] = config.reward['slow']

