import numpy as np
import time
import math
from math import *
from Model.J import J
from Model.Vc import Vc
from Model.WG import WG
from Model.Rudder import Rudder
from Datacode.data_viewer import data_viewer
from Datacode.data_process import data_storage, data_elimation


class Waveglider(object):
    # initialization of data storage lists
    def __init__(self):
        self.n_actions = 5
        self.n_features = 3
        self._t = []
        self.time_step = 1
        # sea state
        self.H = 0.3
        self.omega = 1
        self.c_dir = 0
        self.c_speed = 0
        self.state_0 = np.zeros((8, 1))


        # float
        self.x1 = []
        self.y1 = []
        self.z1 = []
        self.phi1 = []
        self.u1 = []
        self.v1 = []
        self.w1 = []
        self.r1 = []

        # forces
        self.Thrust = []
        self.Rudder_angle = []
        self.Frudder_x = []
        self.Frudder_y = []
        self.Frudder_n = []
        #target position
        self.target_position = np.array([50, 50])
        self.obs_position = np.array([25, 25])
        self.obs_R = 0
        self.safe_distance = self.obs_R

        self.xlim_left = -20
        self.xlim_right = 60
        self.ylim_left = -20
        self.ylim_right = 60

    def reset(self):
        time.sleep(0.1)
        #data_elimation()  # Turn on when previous data needs to be cleared
        self.t = 0
        self._t.clear()
        # float
        self.x1.clear()
        self.y1.clear()
        self.z1.clear()
        self.phi1.clear()
        self.u1.clear()
        self.v1.clear()
        self.w1.clear()
        self.r1.clear()

        # forces
        self.Thrust.clear()

        self.Rudder_angle.clear()
        self.Frudder_x.clear()
        self.Frudder_y.clear()
        self.Frudder_n.clear()
        # initial state
        self.state_0 = np.array([[0], [0], [0], [0],  # eta1
                            [0], [0], [0], [0]],  float)  # V1
        #self.rudder_angle = [0]

        return np.array([self.state_0.item(0), self.state_0.item(1), self.state_0.item(3)])


    def f(self, state, angle):
        #  float's position and attitude vector
        eta1 = state[0:4]
        #eta1[2] = self.H / 2 * sin(self.omega * t)
        WF = np.array([[20], [0], [0], [0]])
        #  float's velocity vector
        V1 = state[4:8]

        #  float's relative velocity vector
        V1_r = V1 - Vc(self.c_dir, self.c_speed, eta1)
        wg = WG(eta1, eta1, V1, V1, self.c_dir, self.c_speed)
        rudder = Rudder(eta1, V1, self.c_dir, self.c_speed)
        # float's kinematic equations
        eta1_dot = np.dot(J(eta1), V1)

        Minv_1 = np.linalg.inv(wg.MRB_1() + wg.MA_1())

        MV1_dot = - np.dot(wg.CRB_1(), V1) - np.dot(wg.CA_1(), V1_r) - np.dot(wg.D_1(), V1_r) - wg.d_1() + rudder.force(angle) + WF

        V1_dot = np.dot(Minv_1, MV1_dot)

        return np.vstack((eta1_dot, V1_dot))

    def change_angle(self, degree):
        if degree > pi:
            output = degree - 2*pi
        elif degree < -pi:
            output = degree + 2*pi
        else:
            output = degree
        return output

    def obser(self, rudder_angle):
        # Runge-Kutta
        k1 = self.f(self.state_0, rudder_angle)
        k2 = self.f(self.state_0 + 0.5 * k1 * self.time_step, rudder_angle)
        k3 = self.f(self.state_0 + 0.5 * k2 * self.time_step, rudder_angle)
        k4 = self.f(self.state_0 + k3 * self.time_step, rudder_angle)
        self.state_0 += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * self.time_step
        self.state_0[3] = self.change_angle(self.state_0.item(3))
        self.t += 1
        #print(self.state_0.item(12))
        self._t.append(self.t)
        self.x1.append(self.state_0.item(0))
        self.y1.append(self.state_0.item(1))
        self.z1.append(self.state_0.item(2))
        self.phi1.append(self.state_0.item(3))
        self.u1.append(self.state_0.item(4))
        self.v1.append(self.state_0.item(5))
        self.w1.append(self.state_0.item(6))
        self.r1.append(self.state_0.item(7))

        self.Rudder_angle.append(rudder_angle)
        # data_storage(self.x1, self.y1, self.phi1, self.t, u1=self.u1, rudder_angle=self.Rudder_angle)  # store data in local files

        observation = np.array([self.state_0.item(0), self.state_0.item(1), self.state_0.item(3)])

        return observation

    def step(self, action):
        '''
        if len(self.Rudder_angle):
            r_a = self.Rudder_angle[-1]
        else:
            r_a = 0

        if r_a > 5*pi/180:
            r_a = 5*pi/180
        elif r_a < -5*pi/180:
            r_a = -5*pi/180
        else:
            r_a = r_a

        a_1 = r_a + 2*pi/180
        a_2 = r_a
        a_3 = r_a - 2*pi/180

        if action == 0:
            s_ = self.obser(a_1)
        elif action == 1:
            s_ = self.obser(a_2)
        elif action == 2:
            s_ = self.obser(a_3)
        '''
        if len(self.x1):
            l_x1 = self.x1[-1]
        else:
            l_x1 = 0
        if len(self.y1):
            l_y1 = self.y1[-1]
        else:
            l_y1 = 0
        l_ps = [l_x1, l_y1]
        s_ = np.array([0, 0, 0])
        a_1 = 5*pi/180
        a_2 = 2*pi/180
        a_3 = 0
        a_4 = -2*pi/180
        a_5 = -5*pi/180

        if action == 0:
            s_ = self.obser(a_1)
        elif action == 1:
            s_ = self.obser(a_2)
        elif action == 2:
            s_ = self.obser(a_3)
        elif action == 3:
            s_ = self.obser(a_4)
        elif action == 4:
            s_ = self.obser(a_5)

        # reward function
        real_position = s_[:2]
        distance_1 = self.target_position - real_position
        distance_2 = self.obs_position - real_position
        distance_3 = self.target_position - l_ps
        distance_4 = self.obs_position - l_ps
        distance_goal = math.hypot(distance_1[0], distance_1[1])
        distance_obs = math.hypot(distance_2[0], distance_2[1]) - self.obs_R
        distance_obs_last = math.hypot(distance_4[0], distance_4[1]) - self.obs_R
        distance_goal_last = math.hypot(distance_3[0], distance_3[1])
        reach = 0

        if distance_goal < 5:
            reward = 100
            reach = 1
            done = True

        elif (s_[0] >= self.target_position[0] + 10 or s_[0] <= -10) or (s_[1] >= self.target_position[1]+10 or s_[1] <= -10):
            reward = -100
            done = True
        elif distance_goal < distance_goal_last:
            reward = -0.5
            done = False
        elif distance_goal > distance_goal_last:
            reward = -1
            done =False
        else:
            reward = 0
            done =False
        return s_, reward, done, reach
        '''
        elif (s_[0] >= self.target_position[0] + 10 or s_[0] <= -10) or (s_[1] >= self.target_position[1]+10 or s_[1] <= -10):
            reward = -100
            done = True
        
        elif distance_obs <= 0:
            reward = -100
            done = True
        elif (distance_obs_last<= self.safe_distance) and (distance_obs<= self.safe_distance):
            reward = -20
            done = False
        elif (distance_obs_last<= self.safe_distance) and (distance_obs> self.safe_distance):
            reward = 10
            done = False
        elif (distance_obs_last> self.safe_distance) and (distance_obs<= self.safe_distance):
            reward = -10
            done = False
        else:
            reward = 100/distance_goal
            done = False
        return s_, reward, done
        '''


    def render(self):
        data_viewer(self.x1, self.y1, u1=self.u1, phit=self.phi1, rudder_angle=self.Rudder_angle, t=self._t, xlim_left=self.xlim_left, xlim_right=self.xlim_right, ylim_left=self.ylim_left, ylim_right=self.ylim_right,
                        goal_x=self.target_position[0], goal_y=self.target_position[1], obs_x=self.obs_position[0], obs_y=self.obs_position[1], R=self.obs_R, s_d=self.safe_distance)