#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import random

class Mdp:

    def __init__(self):
        self.states = [1,2,3,4,5,6,7,8] #状态
        self.terminal_states = dict()
        self.terminal_states[6] = 1
        self.terminal_states[7] =1
        self.terminal_states[8] = 1

        self.actions = ['n','e','s','w'] #北 东 南 西

        self.rewards = dict()#奖励
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        self.t =dict()#状态转移
        self.t['1_s']       = 6
        self.t['1_e']       = 2
        self.t['2_w']       = 1
        self.t['2_e']       = 3
        self.t['3_s']       = 7
        self.t['3_w']       = 2
        self.t['3_e']       = 4
        self.t['4_w']       = 3
        self.t['4_e']       = 5
        self.t['5_s']       = 8
        self.t['5_w']       = 4

        self.gamma = 0.8

    def transform(self, state, action):#return is_terminal, next_state, reward
        if state in self.terminal_states:
            return True, state, 0

        key = '%d_%s'%(state, action)#构成key
        if key in self.t:
            next_state = self.t[key]#可以转移
        else:
            next_state = state#维持原样

        is_terminal = False
        if next_state in self.terminal_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return is_terminal, next_state, r;

