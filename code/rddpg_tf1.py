import os
import csv
import time
import random
import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
import cv2
from airsim_env_tf1 import Env

np.set_printoptions(suppress=True, precision=4)
agent_name = 'rddpg'
num_drone = 3
num_cam = 4

class RDDPGAgent(object):
    
    def __init__(self, state_size, pos_size, action_size, actor_lr, critic_lr, tau,
                gamma, lambd, batch_size, memory_size, 
                epsilon, epsilon_end, decay_step, load_model):
        self.state_size = state_size
        self.pos_size = pos_size
        self.action_size = action_size
        self.action_high = 1.5
        self.action_low = -self.action_high
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        # adjust gpu usage here
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.85
        self.sess = tf.Session(config=tf_config)
        K.set_session(self.sess)

        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.actor_update = self.build_actor_optimizer()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        print("loading model: ", load_model)
        if load_model:
            self.load_model('./save_model/'+ agent_name)
            print("Loaded model for agent.")
        print("done loading")
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)
        print("Done initialize agent.")

    def build_model(self):
        # shared network
        # image process
        image = Input(shape=self.state_size)
        image_process = BatchNormalization()(image)
        image_process = TimeDistributed(
            Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        #72 128
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #70 126
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #68 124
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #34 62
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #32 60
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #30 58
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #15 29
        image_process = TimeDistributed(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #13 27
        image_process = TimeDistributed(Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #10 24
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #5 12
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #3 10
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(48, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)
        
        # vel process
        vel = Input(shape=[self.pos_size])
        vel_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(vel)
        vel_process = BatchNormalization()(vel_process)
        vel_process = Activation('tanh')(vel_process)

        # state process
        state_process = Add()([image_process, vel_process])

        # Actor
        policy1 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy1 = BatchNormalization()(policy1)
        policy1 = ELU()(policy1)
        policy1 = Dense(32, kernel_initializer='he_normal', use_bias=False)(policy1)
        policy1 = BatchNormalization()(policy1)
        policy1 = ELU()(policy1)
        policy1 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy1)
        policy1 = Lambda(lambda x: K.clip(x, self.action_low, self.action_high))(policy1)

        policy2 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy2 = BatchNormalization()(policy2)
        policy2 = ELU()(policy2)
        policy2 = Dense(32, kernel_initializer='he_normal', use_bias=False)(policy2)
        policy2 = BatchNormalization()(policy2)
        policy2 = ELU()(policy2)
        policy2 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy2)
        policy2 = Lambda(lambda x: K.clip(x, self.action_low, self.action_high))(policy2)

        policy3 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy3 = BatchNormalization()(policy3)
        policy3 = ELU()(policy3)
        policy3 = Dense(32, kernel_initializer='he_normal', use_bias=False)(policy3)
        policy3 = BatchNormalization()(policy3)
        policy3 = ELU()(policy3)
        policy3 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy3)
        policy3 = Lambda(lambda x: K.clip(x, self.action_low, self.action_high))(policy3)

        actor = Model(inputs=[image, vel], outputs=[policy1, policy2, policy3])
        
        # Critic
        action1 = Input(shape=[self.action_size])
        action_process1 = Dense(48, kernel_initializer='he_normal', use_bias=False)(action1)
        action_process1 = BatchNormalization()(action_process1)
        action_process1 = Activation('tanh')(action_process1)

        action2 = Input(shape=[self.action_size])
        action_process2 = Dense(48, kernel_initializer='he_normal', use_bias=False)(action2)
        action_process2 = BatchNormalization()(action_process2)
        action_process2 = Activation('tanh')(action_process2)

        action3 = Input(shape=[self.action_size])
        action_process3 = Dense(48, kernel_initializer='he_normal', use_bias=False)(action3)
        action_process3 = BatchNormalization()(action_process3)
        action_process3 = Activation('tanh')(action_process3)

        action_process = Add()([action_process1, action_process2, action_process3])

        state_action = Add()([state_process, action_process])

        Qvalue1 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue1 = BatchNormalization()(Qvalue1)
        Qvalue1 = ELU()(Qvalue1)
        Qvalue1 = Dense(32, kernel_initializer='he_normal', use_bias=False)(Qvalue1)
        Qvalue1 = BatchNormalization()(Qvalue1)
        Qvalue1 = ELU()(Qvalue1)
        Qvalue1 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue1)

        Qvalue2 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue2 = BatchNormalization()(Qvalue2)
        Qvalue2 = ELU()(Qvalue2)
        Qvalue2 = Dense(32, kernel_initializer='he_normal', use_bias=False)(Qvalue2)
        Qvalue2 = BatchNormalization()(Qvalue2)
        Qvalue2 = ELU()(Qvalue2)
        Qvalue2 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue2)

        Qvalue3 = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue3 = BatchNormalization()(Qvalue3)
        Qvalue3 = ELU()(Qvalue3)
        Qvalue3 = Dense(32, kernel_initializer='he_normal', use_bias=False)(Qvalue3)
        Qvalue3 = BatchNormalization()(Qvalue3)
        Qvalue3 = ELU()(Qvalue3)
        Qvalue3 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue3)
        
        critic = Model(inputs=[image, vel, action1, action2, action3], outputs=[Qvalue1, Qvalue2, Qvalue3])

        actor.summary()
        critic.summary()

        actor._make_predict_function()
        critic._make_predict_function()
        
        return actor, critic

    def build_actor_optimizer(self):
        pred_Q1, pred_Q2, pred_Q3 = self.critic.output
        ## split out actions from self.critic.input as self.critic.input consists of all actions (length of 9: action1 + action2 + action3) 
        ##action1, action2, action3 = [self.critic.input[2][x:x+self.action_size] for x in range(0, self.critic.input[2].shape[1], self.action_size)]
        #action1, action2, action3 = tf.split(self.critic.input[2],num_or_size_splits=self.action_size, axis=1)
        #print("action1: ", action1)
        print("pred: ", pred_Q1)
        action_grad1 = tf.gradients(pred_Q1, self.critic.input[2])
        action_grad2 = tf.gradients(pred_Q2, self.critic.input[3])
        action_grad3 = tf.gradients(pred_Q3, self.critic.input[4])
        print("action_grad: ", action_grad1)
        target1 = -action_grad1[0] / self.batch_size
        target2 = -action_grad2[0] / self.batch_size
        target3 = -action_grad3[0] / self.batch_size
        print("target: ", target1)
        params_grad1 = tf.gradients(
            self.actor.output[0], self.actor.trainable_weights, target1)
        params_grad2 = tf.gradients(
            self.actor.output[1], self.actor.trainable_weights, target2)
        params_grad3 = tf.gradients(
            self.actor.output[2], self.actor.trainable_weights, target3)
        
        params_grad = params_grad1 + params_grad2 + params_grad3

        #params_grad1, global_norm1 = tf.clip_by_global_norm(params_grad1, 5.0)
        #params_grad2, global_norm2 = tf.clip_by_global_norm(params_grad2, 5.0)
        #params_grad3, global_norm3 = tf.clip_by_global_norm(params_grad3, 5.0)
        params_grad, global_norm = tf.clip_by_global_norm(params_grad, 5.0)
        grads = zip(params_grad, self.actor.trainable_weights)
        # grads1 = zip(params_grad1, self.actor.trainable_weights)
        # grads2 = zip(params_grad2, self.actor.trainable_weights)
        # grads3 = zip(params_grad3, self.actor.trainable_weights)

        print("params_grad1: ", params_grad)
        print("global_norm1: ", global_norm)
        print("grads1: ", grads)
        #concatglobal_norm = tf.stack([global_norm1, global_norm2, global_norm3], axis=0)
        #global_norm = K.mean(concatglobal_norm)
        #concatgrads = tf.stack([grads], axis=0)

        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        updates = optimizer.apply_gradients(grads)
        train = K.function(
            [self.actor.input[0], self.actor.input[1], self.critic.input[2], self.critic.input[3], self.critic.input[4]],
            [global_norm],
            updates=[updates]
        )
        return train

    def build_critic_optimizer(self):
        y1 = K.placeholder(shape=(None, 1), dtype='float32')
        y2 = K.placeholder(shape=(None, 1), dtype='float32')
        y3 = K.placeholder(shape=(None, 1), dtype='float32')

        pred1, pred2, pred3 = self.critic.output
        
        preloss1 = K.mean(K.square(pred1 - y1))
        preloss2 = K.mean(K.square(pred2 - y2))
        preloss3 = K.mean(K.square(pred3 - y3))

        #concatpreloss = tf.stack([preloss1, preloss2, preloss3], axis=0)
        #print("preloss1: ", preloss1)
        #print("concatpreloss: ", concatpreloss)
        #loss = K.mean(concatpreloss)
        #print("loss: ", loss)
        avgloss = (preloss1 + preloss2 + preloss3)/3
        print("avgloss: ", avgloss)
        # Huber Loss
        # error = K.abs(y - pred)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], avgloss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], self.critic.input[2], self.critic.input[3], self.critic.input[4], y1, y2, y3],
            [avgloss],
            updates=updates
        )
        return train

    def get_action(self, state):
        policy1, policy2, policy3 = self.actor.predict(state)
        policy1, policy2, policy3 = policy1[0], policy2[0], policy3[0]
        noise1 = np.random.normal(0, self.epsilon, self.action_size)
        noise2 = np.random.normal(0, self.epsilon, self.action_size)
        noise3 = np.random.normal(0, self.epsilon, self.action_size)
        action1 = np.clip(policy1 + noise1, self.action_low, self.action_high)
        action2 = np.clip(policy2 + noise2, self.action_low, self.action_high)
        action3 = np.clip(policy3 + noise3, self.action_low, self.action_high)
        return [(action1, policy1), (action2, policy2), (action3, policy3)]

    def train_model(self):
        print(f'lem mem: {len(self.memory)}, batch: {self.batch_size}')
        batch = random.sample(self.memory, self.batch_size)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.pos_size])

        actions1 = np.zeros((self.batch_size, self.action_size))
        actions2 = np.zeros((self.batch_size, self.action_size))
        actions3 = np.zeros((self.batch_size, self.action_size))

        rewards = np.zeros((self.batch_size, 1))

        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.pos_size])

        dones = np.zeros((self.batch_size, 1))

        targets = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):
            images[i], vels[i] = sample[0]
            actions1[i] = sample[1]
            actions2[i] = sample[2]
            actions3[i] = sample[3]
            rewards[i] = sample[4]
            next_images[i], next_vels[i] = sample[5]
            dones[i] = sample[6]
        states = [images, vels]
        next_states = [next_images, next_vels]
        policy1, policy2, policy3 = self.actor.predict(states)
        target_actions1, target_actions2, target_actions3 = self.target_actor.predict(next_states)
        #target_actions = np.concatenate([target_actions1,target_actions2,target_actions3], axis=1)
        target_next_Qs1, target_next_Qs2, target_next_Qs3 = self.target_critic.predict(next_states + [target_actions1, target_actions2, target_actions3])
        targets1 = rewards + self.gamma * (1 - dones) * target_next_Qs1
        targets2 = rewards + self.gamma * (1 - dones) * target_next_Qs2
        targets3 = rewards + self.gamma * (1 - dones) * target_next_Qs3

        #policy = np.hstack((policy1, policy2, policy3))
        #actions = np.hstack((actions1, actions2, actions3))
        actor_loss = self.actor_update(states + [policy1, policy2, policy3])
        critic_loss = self.critic_update(states + [actions1, actions2, actions3, targets1, targets2, targets3])
        return actor_loss[0], critic_loss[0]

    def append_memory(self, state, action1, action2, action3, reward, next_state, done):        
        self.memory.append((state, action1, action2, action3, reward, next_state, done))
        
    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded: ', name + '_actor.h5')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded:', name + '_critic.h5')

    def save_model(self, name):
        self.actor.save_weights(name + '_actor.h5')
        self.critic.save_weights(name + '_critic.h5')

    def update_target_model(self):
        self.target_actor.set_weights(
            self.tau * np.array(self.actor.get_weights()) \
            + (1 - self.tau) * np.array(self.target_actor.get_weights())
        )
        self.target_critic.set_weights(
            self.tau * np.array(self.critic.get_weights()) \
            + (1 - self.tau) * np.array(self.target_critic.get_weights())
        )


'''
Environment interaction
'''

def transform_input(responses, img_height, img_width):
    dimg_list = []
    for img in responses:
        # resize the image to half, from (224, 352) to (112, 176), so that less parameter is needed for the networks.
        img_resized = cv2.resize(img, (img_width, img_height))
        dimg = np.array(cv2.cvtColor(img_resized[:,:,:3], cv2.COLOR_BGR2GRAY))
        dnorm = np.zeros((img_height, img_width))
        dnorm = cv2.normalize(dimg, dnorm, 0, 255, cv2.NORM_MINMAX)
        dimg_list.append(dnorm)
    dimg_all = np.array(dimg_list)
    #print("dimg_all: ", dimg_all.shape)
    image = dimg_all.reshape(1, img_height, img_width, len(responses))
    print("transform responses len: ", len(responses))
    #cv2.imwrite('view.png', dimg)
    return image

def transform_action(action):
    real_action = np.array(action)
    real_action[1] += 0.5
    return real_action

if __name__ == '__main__':
    # CUDA config
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play',       action='store_true')
    # img_height: 112, img_width: 176
    parser.add_argument('--img_height', type=int,   default=112)
    parser.add_argument('--img_width',  type=int,   default=176)
    parser.add_argument('--actor_lr',   type=float, default=1e-4)
    parser.add_argument('--critic_lr',  type=float, default=5e-4)
    parser.add_argument('--tau',        type=float, default=5e-3)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--lambd',      type=float, default=0.90)
    parser.add_argument('--seqsize',    type=int,   default=5)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--memory_size',type=int,   default=10000)
    parser.add_argument('--train_start',type=int,   default=200)
    parser.add_argument('--train_rate', type=int,   default=5)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.05)
    parser.add_argument('--decay_step', type=int,   default=20000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name):
        os.makedirs('save_graph/'+ agent_name)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent
    state_size = [args.seqsize, args.img_height, args.img_width, num_drone*num_cam]
    action_size = 3
    pos_size = num_drone*3
    agent = RDDPGAgent(
        state_size=state_size,
        pos_size=pos_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )

    episode = 0
    env = Env()

    if args.play:
        print("Evaluation process")
        while True:
            try:
                done = False
                bug = False

                # stats
                # stats
                bestReward, timestep, score, avgvel, avgQ, avgAct = 0., 0, 0., 0., 0., 0.

                observe = env.reset()
                image, vel = observe
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                print(f'Main Loop: done: {done}, timestep: {timestep}')
                while not done:
                    print(f'Sub Loop: timestep: {timestep}')
                    timestep += 1
                    # snapshot = np.zeros([0, args.img_width, 1])
                    # for snap in state[0][0]:
                    #     snapshot = np.append(snapshot, snap, axis=0)
                    # snapshot *= 128
                    # snapshot += 128
                    # cv2.imshow('%s' % timestep, np.uint8(snapshot))
                    # cv2.waitKey(0)
                    action1, action2, action3 = agent.actor.predict(state)
                    print("check1 action1:", action1)
                    action1, action2, action3 = action1[0], action2[0], action3[0]
                    print("check2 action1:", action1)
                    noise = [np.random.normal(scale=args.epsilon) for _ in range(action_size)]
                    noise = np.array(noise, dtype=np.float32)
                    action1 = np.clip(action1 + noise, -1, 1)
                    action2 = np.clip(action2 + noise, -1, 1)
                    action3 = np.clip(action3 + noise, -1, 1)
                    real_action1 = transform_action(action1)
                    real_action2 = transform_action(action2)
                    real_action3 = transform_action(action3)
                    observe, reward, done, info = env.step([transform_action(real_action1), transform_action(real_action2), transform_action(real_action3)])
                    image, vel = observe
                    try:
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]
                    reward = np.sum(np.array(reward))
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']

                    # stats
                    #action = np.concatenate([action1,action2,action3])
                    #print("action: ", action)
                    Qs1, Qs2, Qs3 = agent.critic.predict([state[0], state[1], action1.reshape(1, -1), action2.reshape(1, -1), action3.reshape(1, -1)])
                    Qs1, Qs2, Qs3 = Qs1[0][0], Qs2[0][0], Qs3[0][0]
                    avgQ += float(Qs1 + Qs2 + Qs3)
                    avgvel += float(np.linalg.norm(real_action1)+np.linalg.norm(real_action2)+np.linalg.norm(real_action3))
                    score += float(reward)
                    if float(reward) > bestReward:
                        bestReward = float(reward)
                    print('%s' % (real_action1), end='\r', flush=True)
                    print('%s' % (real_action2), end='\r', flush=True)
                    print('%s' % (real_action3), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

                    state = next_state

                if bug:
                    continue
                
                avgQ /= timestep
                avgvel /= timestep

                # done
                print('Ep %d: BestReward %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f Info1 %s Info2 %s Info3 %s'
                        % (episode, bestReward, timestep, score, avgQ, avgvel, info1, info2, info3))

                stats = [
                    episode, timestep, score, bestReward, avgvel, \
                    avgQ, avgAct, info[0]['status'], info[1]['status'], info[2]['status']
                ]
                # log stats
                with open('save_stat/'+ agent_name + '_test_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        print("Training process")
        time_limit = 600
        highscore = -9999999999.
        if os.path.exists('save_stat/'+ agent_name + '_stat.csv'):
            with open('save_stat/'+ agent_name + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name + '_highscore.csv'):
            with open('save_stat/'+ agent_name + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscore = float(next(reversed(list(read)))[0])
                print('Highscore:', highscore)
        global_step = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestReward, timestep, score, avgvel, avgQ, avgAct = 0., 0, 0., 0., 0., 0.
                train_num, actor_loss, critic_loss = 0, 0., 0.

                observe = env.reset()
                image, vel = observe
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    print("transform_image error..")
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                print(f'Main Loop: done: {done}, timestep: {timestep}, time_limit: {time_limit}')
                while not done and timestep < time_limit:
                    print(f'Sub Loop: timestep: {timestep}, global_step: {global_step}')
                    timestep += 1
                    global_step += 1
                    print("len(agent.memory): ", len(agent.memory))
                    print("args.train_start: ", args.train_start)
                    print("args.train_rate: ", args.train_rate)
                    if len(agent.memory) >= args.train_start and global_step >= args.train_rate:
                        print('Training model')
                        for _ in range(args.epoch):
                            a_loss, c_loss = agent.train_model()
                            actor_loss += float(a_loss)
                            critic_loss += float(c_loss)
                            train_num += 1
                        agent.update_target_model()
                        global_step = 0
                    (action1, policy1), (action2, policy2), (action3, policy3) = agent.get_action(state)
                    #print("get_action results: ", agent.get_action(state))
                    real_action1, real_policy1 = transform_action(action1), transform_action(policy1)
                    real_action2, real_policy2 = transform_action(action2), transform_action(policy2)
                    real_action3, real_policy3 = transform_action(action3), transform_action(policy3)
                    observe, reward, done, info = env.step([real_action1, real_action2, real_action3])
                    image, vel = observe
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']
                    print("Done: ", done)
                    print("Timestep: ", timestep)
                    try:
                        print("STATUS: ", timestep, info[0]['status'], info[1]['status'], info[2]['status'])
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]
                    reward = np.sum(np.array(reward))
                    agent.append_memory(state, action1, action2, action3, reward, next_state, done)

                    # stats
                    action = np.concatenate([action1,action2,action3])
                    print("action: ", action)
                    Qs1, Qs2, Qs3 = agent.critic.predict([state[0], state[1], action1.reshape(1, -1), action2.reshape(1, -1), action3.reshape(1, -1)])
                    Qs1, Qs2, Qs3 = Qs1[0][0], Qs2[0][0], Qs3[0][0]
                    avgQ += float(Qs1 + Qs2 + Qs3)
                    avgvel += float(np.linalg.norm(real_policy1)+np.linalg.norm(real_policy2)+np.linalg.norm(real_policy3))
                    avgAct += float(np.linalg.norm(real_action1)+np.linalg.norm(real_action2)+np.linalg.norm(real_action3))
                    score += float(reward)
                    if float(reward) > bestReward:
                        bestReward = float(reward)
                    print('%s | %s' % (real_action1, real_policy1))
                    print('%s | %s' % (real_action2, real_policy2))
                    print('%s | %s' % (real_action3, real_policy3))

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

                    state = next_state

                    if agent.epsilon > agent.epsilon_end:
                        agent.epsilon -= agent.epsilon_decay

                if bug:
                    continue

                if train_num:
                    actor_loss /= train_num
                    critic_loss /= train_num

                avgQ /= timestep
                avgvel /= timestep
                avgAct /= timestep

                # done
                if args.verbose:
                    print('Ep %d: BestReward %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f AvgAct %.2f Info1 %s Info2 %s Info3 %s'
                        % (episode, bestReward, timestep, score, avgQ, avgvel, avgAct, info1, info2, info3))

                stats = [
                    episode, timestep, score, bestReward, avgvel, \
                    actor_loss, critic_loss, avgQ, avgAct, info[0]['status'], info[1]['status'], info[2]['status']
                ]
                # log stats
                with open('save_stat/'+ agent_name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])
                if highscore < bestReward:
                    highscore = bestReward
                    with open('save_stat/'+ agent_name + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscore, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    agent.save_model('./save_model/'+ agent_name + '_best')
                agent.save_model('./save_model/'+ agent_name)
                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break