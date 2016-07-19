#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import random as ran
ran.seed(0)
from mdp import *

#Get next action
def random_pi():
    actions = ['n','w','e','s']
    index = int(ran.random()*4)
    return actions[index]

#Compute state value
def compute_random_pi_state_value():
    value = [0.0 for r in xrange(9)]
    num = 1000000

    for k in xrange(1,num):
        for i in xrange(1,6):
            mdp = Mdp()
            s = i;
            is_terminal = False
            gamma = 1.0
            v = 0.0
            #start algorithm
            while False == is_terminal:
                a = random_pi()
                is_terminal, s, r = mdp.transform(s,a)
                v += gamma*r
                gamma *= 0.5

            value[i] = (value[i] * (k-1) + v) / k

        if k % 100000 == 0:
            print value

    print value

compute_random_pi_state_value()

