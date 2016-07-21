#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import random
random.seed(0)

def update(policy, s_fea, a, tvalue, alpha):
    pvalue = policy.qfunc(s_fea, a)
    error = pvalue - tvalue
    s_a_fea = policy.get_state_action_fea(s_fea, a)
    policy.theta -= alpha * error * s_a_fea


def qlearning(env, policy, num_iter1, alpha, gamma):
    actions = policy.actions
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        s_f = env.reset() #reset environment
        a = policy.epsilon_greedy(s_f)
        count = 0
        t =False
        while False == t and count < 10000:
            s_f1,r,t,i = env.step(a) #step
            qmax = policy.qfunc(s_f1, a)
            for a1 in policy.actions:
                pvalue = policy.qfunc(s_f1, a1)
                if qmax < pvalue:
                    qmax = pvalue
            update(policy, s_f, a, r + gamma * qmax, alpha)
            s_f = s_f1
            a = policy.epsilon_greedy(s_f)
            count += 1

        if (iter1%100) == 0:
            print "complete the %d epoches" % (iter1)

    return policy