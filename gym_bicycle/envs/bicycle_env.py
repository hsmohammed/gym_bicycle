# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:47:48 2019

@author: Hossameldin Mohammed
"""
import numpy as np
#import matplotlib.pyplot as plt
#import math
import gym
from gym import spaces
from gym.utils import seeding



class BicycleEnv(gym.Env):
    def __init__(self):
        self.min_speed = 0.006018707
        self.min_speedDiff = -4.944894
        self.min_longDistance = 0.001033224
        self.min_latDistance = -1.740935
        self.min_pathDeviation = -1.249996
        self.min_dirAngle = -49.47395
        
        self.max_speed = 6.392389
        self.max_speedDiff = 2.815963
        self.max_longDistance = 23.57297
        self.max_latDistance = 2.112076
        self.max_pathDeviation = 1.128726
        self.max_dirAngle = 53.47108
        
        self.min_yawRate = -75.6378
        self.min_Acc = -9.841724
        
        self.max_yawRate = 72.07896
        self.max_Acc = 10.50623
        
        self.min_x = 6.5
        self.max_x = 9
        
        self.min_y = -25
        self.max_y = -4
        self.viewer = None
        
        self.timeStep = 1/30
        
        self.low_state = np.array([self.min_speed,
                                   self.min_speedDiff,
                                   self.min_longDistance,
                                   self.min_latDistance,
                                   self.min_pathDeviation,
                                   self.min_dirAngle,
                                   self.min_x,
                                   self.min_y])
    
        self.high_state = np.array([self.max_speed,
                                   self.max_speedDiff,
                                   self.max_longDistance,
                                   self.max_latDistance,
                                   self.max_pathDeviation,
                                   self.max_dirAngle,
                                   self.max_x,
                                   self.max_y])

        
        self.action_space = spaces.Box(np.array([self.min_yawRate,self.min_Acc]),
                                       np.array([self.max_yawRate,self.max_Acc]))
        
        self.observation_space = spaces.Box(np.array([self.min_speed,self.min_speedDiff, self.min_longDistance,
                                                      self.min_latDistance, self.min_pathDeviation, self.min_dirAngle, self.min_x, self.min_y]), 
                                            np.array([self.max_speed,self.max_speedDiff, self.max_longDistance,
                                                      self.max_latDistance, self.max_pathDeviation, self.max_dirAngle, self.min_x, self.min_y]))
    
        self.seed()
        self.reset()
        
        
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):

        t = self.timeStep
        
        speed = self.state[0]
        speedDiff = self.state[1]
        longDistance = self.state[2]
        latDistance = self.state[3]
        pathDeviation = self.state[4]
        dirAngle = self.state[5]
        x = self.state[6]
        y = self.state[7]
        
        yawRate = action[0]
        acc = action[1]
        
        speed_lead = speed - speedDiff
        dirAngle_lead = 0
        acc_lead = 0
        distance_lead = speed_lead * t + 0.5 * acc_lead * ((t)**2)
        dx_lead = distance_lead * np.sin(dirAngle_lead)
        dy_lead = distance_lead * np.cos(dirAngle_lead)

        distance = speed * t + 0.5 * acc * ((t)**2)
        
        dirAngle += yawRate * t
        
        dx = distance * np.sin(dirAngle)
        dy = distance * np.cos(dirAngle)
        
        if ((x + dx) > self.max_x):
            dx = self.max_x - x
        elif ((x + dx) < self.min_x):
            dx = self.min_x - x
            
        speed += acc * t
        pathDeviation += dx
        
        longDistance += dy_lead - dy
        latDistance += dx - dx_lead
        speedDiff += acc * t
        
        x += dx
        y+= dy
        
        reward = -10 * speed - 1 * speedDiff + 15 * longDistance + 0.5 * latDistance - 20 * pathDeviation - 1 * dirAngle
        
        self.state = np.array([speed, speedDiff, longDistance, latDistance, pathDeviation, dirAngle, x, y])
        
        done = bool(y > self.max_y)
        
        
        return self.state, reward, done, {}

        
        
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=self.min_speed, high=self.max_speed),
                               self.np_random.uniform(low=self.min_speedDiff, high=self.max_speedDiff),
                               self.np_random.uniform(low=self.min_longDistance, high=self.max_longDistance),
                               self.np_random.uniform(low=self.min_latDistance, high=self.max_latDistance),
                               self.np_random.uniform(low=self.min_pathDeviation, high=self.max_pathDeviation),
                               self.np_random.uniform(low=self.min_dirAngle, high=self.max_dirAngle),
                               self.np_random.uniform(low=self.min_x, high=self.max_x),
                               -25])
        return np.array(self.state)
    
    
        
    def _height(self, xs):
        return np.full((20,1),15)

        
#    def render(self, mode = 'human'):
#        screen_width = 250
#        screen_height = 2100
#
#        world_width = self.max_x - self.min_x
#        scale = screen_width/world_width
#        bike_width=60
#        bike_length=175
#        
#        
#        if self.viewer is None:
#            from gym.envs.classic_control import rendering
#            
#            self.viewer = rendering.Viewer(screen_width, screen_height)
#            xs = np.linspace(self.min_x, self.max_x, 20)
#            ys = self._height(xs)
#            xys = list(zip((xs - self.min_x)*scale, ys*scale))
##            
#            self.track = rendering.make_polyline(xys)
#            self.track.set_linewidth(1)
#            self.viewer.add_geom(self.track)
#            
##            clearance = 10
###            
##            l,r,t,b = -bike_width/2, bike_width/2, bike_length, -10
##            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
##            car.add_attr(rendering.Transform(translation=(0, clearance)))
##            self.cartrans = rendering.Transform()
##            car.add_attr(self.cartrans)
##            self.viewer.add_geom(car)
##            frontwheel = rendering.make_circle(bike_length/2.5)
##            frontwheel.set_color(.5, .5, .5)
##            frontwheel.add_attr(rendering.Transform(translation=(bike_width/4,clearance)))
##            frontwheel.add_attr(self.cartrans)
##            self.viewer.add_geom(frontwheel)
##            backwheel = rendering.make_circle(bike_length/2.5)
##            backwheel.add_attr(rendering.Transform(translation=(-bike_width/4,clearance)))
##            backwheel.add_attr(self.cartrans)
##            backwheel.set_color(.5, .5, .5)
##            self.viewer.add_geom(backwheel)
##            flagx = (7-self.min_x)*scale
##            flagy1 = self._height(7)*scale
##            flagy2 = flagy1 + 50
##            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
##            self.viewer.add_geom(flagpole)
##            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
##            flag.set_color(.8,.8,0)
##            self.viewer.add_geom(flag)
#
##        pos = 6.5
##        self.cartrans.set_translation((pos-self.min_x)*scale, self._height(pos)*scale)
##        self.cartrans.set_rotation(math.cos(3 * pos))
###        
##        mode = 'human'
#
#        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
                        
            
if __name__ == '__main__':
    env = BikePath()
#    env.render()


        
