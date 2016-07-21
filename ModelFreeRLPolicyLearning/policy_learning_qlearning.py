#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import random
random.seed(0)
import grid_mdp
import matplotlib.pyplot as plt

grid = grid_mdp.Grid_Mdp()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()

#epsilon greedy policy#
def epsilon_greedy(qfunc, state, epsiloon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = qfunc[key]
    for i in xrange(len(actions)):
        key = "%d_%s"%(state, actions[i])
        if qmax < qfunc[key]:
            qmax = qfunc[key]
            amax = i

    #probability
    pro = [0.0 for i in xrange(len(actions))]
    pro[amax] += 1 - epsiloon
    for i in xrange(len(pro)):
        pro[i] += epsiloon / len(actions)

    #choose
    r = random.random()
    s = 0.0
    for i in xrange(len(actions)):
        s += pro[i]
        if s >= r: return actions[i]
    return actions[len(actions)-1]

best = dict()
def read_best():
    f = open("best_qfunc")
    for line in f:
        line = line.strip()
        if len(line) == 0: continue
        element = line.split(":")
        best[element[0]] = float(element[1])

def compute_error(qfunc):
    sum1 = 0.0
    for key in qfunc:
        error = qfunc[key] - best[key]
        sum1 += error * error
    return  sum1

def qlearning(num_iter1, alpha, epsilon):
    x = []
    y = []
    qfunc = dict()
    for s in states:
        for a in actions:
            key = "%d_%s"%(s, a)
            qfunc[key] = 0.0

    for iter1 in xrange(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))

        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0
        while False == t and count < 100:
            key = "%d_%s"%(s, a)
            t, s1, r = grid.transform(s, a)

            #find max qfunc at s1
            qmax = -1;
            for a1 in actions:
                key1 = "%d_%s"%(s1, a1)
                if qmax < qfunc[key1]:
                    qmax = qfunc[key1]
            #update
            qfunc[key] = qfunc[key] + alpha * (r + gamma * qmax - qfunc[key])
            s = s1
            a = epsilon_greedy(qfunc, s1, epsilon)
            count += 1

    plt.plot(x,y,"-.,",label="q alpha=%2.1f epsilon=%2.1f"%(alpha, epsilon))
    plt.show(True)
    return qfunc

if __name__ == "__main__":
    read_best()
    qfunc = qlearning(1000, 0.2, 0.2)
    print ""
    print "qlearing_onLine_policy"
    for s in states:
        for a in actions:
            print "%d_%s:%f"%(s,a,qfunc["%d_%s"%(s,a)])