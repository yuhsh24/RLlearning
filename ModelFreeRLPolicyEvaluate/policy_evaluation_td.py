#!/usr/bin/python
# -*- coding: UTF-8 -*-

import grid_mdp
import random

grid = grid_mdp.Grid_Mdp()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()

def td(alpha, gamma, state_sample, action_sample, reward_sample):
    vfunc = dict()
    for s in states:
        vfunc[s] = random.random()

    for iter1 in xrange(len(state_sample)):
        for step in xrange(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            r = reward_sample[iter1][step]

            if len(state_sample[iter1])-1 > step:
                s1 = state_sample[iter1][step+1]
                next_v = vfunc[s1]
            else:
                next_v = 0.0

            vfunc[s] = vfunc[s] + alpha * (r + gamma * next_v - vfunc[s])

    return vfunc

if __name__ == "__main__":
    s, a, r = grid.gen_randompi_sample(1000000)
    value = td(0.05, 0.5, s, a, r)
    print "Time difference value evaluation:"
    for state in xrange(1,6):
        print "%d:%f\t"%(state, value[state])
    print ""