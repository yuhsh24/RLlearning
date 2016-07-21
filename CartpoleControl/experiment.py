#!/usr/bin/python
# -*- coding:UTF-8 -*-

import random
random.seed(10)
from algorithms import *
from mdp import *
from policy import *

if __name__ == "__main__":
    cartpole = get_env("CartPole-v0")
    policy = Policy(cartpole, epsilon = 0.01)
    policy = qlearning(cartpole, policy, \
                       num_iter1 = 10000,\
                       alpha = 0.1, \
                       gamma = 0.9)

    cartpole.monitor.start('./cartpole-experiment-1',force=True)
    for iter1 in xrange(20):
        s_f = cartpole.reset()
        for iter2 in xrange(2000):
            cartpole.render()
            a = policy.greedy(s_f)
            s_f, r, t, i = cartpole.step(a)
            if t:
                print "Episode finished after {} timesteps".format(iter2+1)
                break
    cartpole.monito.close()
