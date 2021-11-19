import time
import csv
import math
import argparse
import numpy as np
from datetime import datetime as dt
from airsim_env_tf1 import Env, ACTION

num_drone = 3
agent_name = "random"

class RandomAgentDiscrete(object):

    def __init__(self, action_size):
        self.action_size = action_size
        self.name = agent_name + "_d"

    def get_action(self):
        actions = []
        for i in range(num_drone):
            action = np.random.choice(self.action_size)
            actions.append(action)
        return actions


class RandomAgentContinuous(object):

    def __init__(self, action_size):
        self.action_size = action_size
        self.name = agent_name + "_c"

    def get_action(self):
        actions = []
        for i in range(num_drone):
            action = np.random.uniform(-1.5, 1.5, self.action_size)
            actions.append(action)
        return actions


def interpret_action(action):
    scaling_factor = 1.0
    if action == 0:
        # quad_offset: (x, y, z, action_type)
        quad_offset = (0, 0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0, 1)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0, 2)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor, 3)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0, 4)
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0, 5)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor, 6)

    return quad_offset


if __name__ == '__main__':
    time_limit = 999
    highscore = -9999999999.
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--continuous', action='store_true')
    args = parser.parse_args()

    if args.continuous:
        print("RandomAgentContinuous")
        agent = RandomAgentContinuous(3)
    else:
        print("RandomAgentDiscrete")
        agent = RandomAgentDiscrete(7)
    env = Env()

    episode = 0
    
    while True:
        done = False
        bestReward = 0
        timestep = 0
        score = 0
        _ = env.reset()
        print(f'Main Loop: done: {done}, timestep: {timestep}, time_limit: {time_limit}')
        while not done and timestep < time_limit:
            print(f'Sub Loop: timestep: {timestep}')
            timestep += 1
            actions = agent.get_action()
            action1, action2, action3 = actions[0], actions[1], actions[2]
            if not args.continuous:
                # if discrete then do interpret action
                real_action1, real_action2, real_action3 = interpret_action(action1), interpret_action(action2), interpret_action(action3)
                print('ACTION: %s' % (ACTION[action1]))
                print('ACTION: %s' % (ACTION[action2]))
                print('ACTION: %s' % (ACTION[action3]))
            else:
                real_action1, real_action2, real_action3 = action1, action2, action3
                print('ACTION: %s' % (action1))
                print('ACTION: %s' % (action2))
                print('ACTION: %s' % (action3))
            _, reward, done, info = env.step([real_action1, real_action2, real_action3])
            info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']
            print("Done: ", done)
            print("Timestep: ", timestep)

            reward = np.sum(np.array(reward))
            score += float(reward)
            if float(reward) > bestReward:
                bestReward = float(reward)

            

            # stack history here
            if args.verbose:
                print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))
        # done
        print('Ep %d: BestReward %.3f Step %d Score %.3f' % (episode, bestReward, timestep, score))
        stats = [
                episode, timestep, score, bestReward, \
                info[0]['status'], info[1]['status'], info[2]['status']
            ]

        # log stats
        with open('save_stat/'+ agent.name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])
        if highscore < bestReward:
            highscore = bestReward
            with open('save_stat/'+ agent.name + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow('%.4f' % s if type(s) is float else s for s in [highscore, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])    
        episode += 1

