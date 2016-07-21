#!/usr/bin/python
# -*- coding:UTF-8 -*-

import grid_mdp
import policy
import evaluate
import numpy as np
import random
random.seed(0)

#update strategy : stochastic gradient
def update(policy, f, a, tvalue, alpha):
    pvalue = policy.qfunc(f, a)
    error = pvalue - tvalue
    fea = policy.get_fea_vec(f, a)
    policy.theta -= alpha * error * fea


#mc value approximation
def mc(grid, policy, evaler, num_iter1, alpha):
    actions = grid.actions
    gamma = grid.gamma
    y = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        y.append(evaler.eval(policy))
        #generate sample
        a_sample = []
        f_sample = []
        r_sample = []

        f = grid.start()
        t = False
        count = 0
        while False == t and count < 100:
            a = policy.epsilon_greedy(f)
            t, f1, r = grid.receive(a)
            a_sample.append(a)
            f_sample.append(f)
            r_sample.append(r)
            f = f1
            count += 1

        g = 0.0
        for i in xrange(len(f_sample)-1,-1, -1):
            g *= gamma
            g += r_sample[i]

        for i in xrange(len(f_sample)):
            update(policy, f_sample[i], a_sample[i], g, alpha)
            g -= r_sample[i]
            g /= gamma

    return policy,y

#sarsa value approximation
def sarsa(grid, policy, evaler, num_iter1, alpha):
    actions = grid.actions
    gamma = grid.gamma
    y = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        y.append(evaler.eval(policy))
        f = grid.start()
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0
        while False == t and count < 100:
            t, f1, r = grid.receive(a)
            a1 = policy.epsilon_greedy(f1)
            update(policy, f, a, r + gamma * policy.qfunc(f1, a1), alpha)
            f = f1
            a = a1
            count += 1

    return policy,y

#qlearning value approximation
def qlearning(grid, policy, evaler, num_iter1, alpha):
    actions = grid.actions
    gamma = grid.gamma
    y = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        y.append(evaler.eval(policy))
        f = grid.start()
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0
        while False == t and count < 100:
            t, f1, r = grid.receive(a)
            #find max Q
            qmax = policy.qfunc(f1, actions[0])
            for a1 in actions:
                pvalue = policy.qfunc(f1, a1)
                if qmax < pvalue:
                    qmax = pvalue
            update(policy, f, a, r + gamma * qmax, alpha)
            f = f1
            a = policy.epsilon_greedy(f1)
            count += 1

    return policy,y