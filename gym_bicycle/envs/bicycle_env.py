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
        
        
        
        self.min_speed_follow = 0
        self.min_speed_lead = 0

        self.min_dirAngle_follow = -55
        self.min_dirAngle_lead = -55
        
        self.min_x_follow = 6.5
        self.min_x_lead = 6.5

        self.min_y_follow = 0
        self.min_y_lead = 0

        self.max_speed_follow = 6.5
        self.max_speed_lead = 6.5

        self.max_dirAngle_follow = 55
        self.max_dirAngle_lead = 55
        
        self.max_x_follow = 9
        self.max_x_lead = 9

        self.max_y_follow = 21
        self.max_y_lead = 42

        
        self.min_yawRate_follow = -80
        self.min_yawRate_lead = -80

        self.min_Acc_follow = -11
        self.min_Acc_lead = -11

        
        self.max_yawRate_follow = 80
        self.max_yawRate_lead = 80

        self.max_Acc_follow = 11
        self.max_Acc_lead = 11


        

        
        self.viewer = None
        
        self.timeStep = 1/30
        
        self.low_state = np.array([self.min_speed_follow, self.min_speed_lead,
                                   self.min_dirAngle_follow, self.min_dirAngle_lead,
                                   self.min_x_follow, self.min_x_lead,
                                   self.min_y_follow, self.min_y_lead])
    
        self.high_state = np.array([self.max_speed_follow, self.max_speed_lead,
                                   self.max_dirAngle_follow, self.max_dirAngle_lead,
                                   self.max_x_follow, self.max_x_lead,
                                   self.max_y_follow, self.max_y_lead])

        
        self.action_space = spaces.Box(np.array([self.min_yawRate_follow, 
                                                 self.min_Acc_follow]),
                                       np.array([self.min_yawRate_follow, 
                                                 self.min_Acc_follow]))
        
        self.observation_space = spaces.Box(np.array([self.min_speed_follow, self.min_speed_lead,
                                   self.min_dirAngle_follow, self.min_dirAngle_lead,
                                   self.min_x_follow, self.min_x_lead,
                                   self.min_y_follow, self.min_y_lead]), 
                                            np.array([self.max_speed_follow, self.max_speed_lead,
                                   self.max_dirAngle_follow, self.max_dirAngle_lead,
                                   self.max_x_follow, self.max_x_lead,
                                   self.max_y_follow, self.max_y_lead]))
    
        self.seed()
        self.reset()
        
        
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):

        t = self.timeStep
        
        speed_follow = self.state[0]
        speed_lead = self.state[1]
        dirAngle_follow = self.state[2]
        dirAngle_lead =  self.state[3]
        x_follow = self.state[4]
        x_lead = self.state[5]
        y_follow = self.state[6]
        y_lead = self.state[7]
        
        pathDeviation_follow = x_follow - 7.75
        
        longDistance = y_lead - y_follow
        latDistance = x_lead - x_follow
        dirAngleDiff = dirAngle_lead - dirAngle_follow
        speedDiff = speed_lead - speed_follow
        

        
        yawRate_follow = action[0]
        acc_follow = action[1]
        
        acc_lead = 0
        yawRate_lead = 0
        
        distance_lead = speed_lead * t + 0.5 * acc_lead * ((t)**2)
        dirAngle_lead += yawRate_lead * t
        dx_lead = distance_lead * np.sin(dirAngle_lead)
        dy_lead = distance_lead * np.cos(dirAngle_lead)

        distance_follow = speed_follow * t + 0.5 * acc_follow * ((t)**2)
        dirAngle_follow += yawRate_follow * t
        dx_follow = distance_follow * np.sin(dirAngle_follow)
        dy_follow = distance_follow * np.cos(dirAngle_follow)
        
        if ((x_follow + dx_follow) > self.max_x_follow):
            dx_follow = self.max_x_follow - x_follow
        elif ((x_follow + dx_follow) < self.min_x_follow):
            dx_follow = self.min_x_follow - x_follow
            
        speed_follow += acc_follow * t
        speed_lead += acc_lead * t
        pathDeviation_follow += dx_follow
        
        longDistance += dy_lead - dy_follow
        latDistance += dx_lead - dx_follow
        dirAngleDiff =  dirAngle_lead - dirAngle_follow
        speedDiff = speed_lead - speed_follow
        
        x_follow += dx_follow
        x_lead += dx_lead

        y_follow += dy_follow
        y_lead += dy_lead

        
        reward = -10 * speed_follow - 1 * speedDiff + 15 * longDistance + 0.5 * latDistance - 20 * pathDeviation_follow - 1 * dirAngle_follow - 1 * dirAngleDiff
        
        self.state = np.array([speed_follow, speed_lead, dirAngle_follow, dirAngle_lead, x_follow, x_lead, y_follow, y_lead])
        
        done = bool(y_follow > self.max_y_follow)
        
        
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
    env = BicycleEnv()
#    env.render()


        
