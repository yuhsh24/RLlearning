#!/usr/bin/python
# -*- coding: UTF-8 -*-

import grid_mdp
import random
random.seed(0)
import matplotlib.pyplot as plt

grid = grid_mdp.Grid_Mdp()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()

#epsilon greedy policy#
def epsilon_greedy(qfunc, state, epsilon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = qfunc[key]
    for i in xrange(len(actions)):
        key = "%d_%s"%(state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i

    #probability
    pro = [0.0 for i in xrange(len(actions))]
    pro[amax] += 1 - epsilon
    for i in xrange(len(actions)):
        pro[i] += epsilon / len(actions)

    #choose
    r = random.random()
    s = 0.0
    for i in xrange(len(actions)):
        s += pro[i]
        if s >= r: return actions[i]
    return actions[len(actions) - 1]

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

def sarsa(num_iter1, alpha, epsilon):
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
            key = "%d_%s"%(s,a)
            t, s1, r = grid.transform(s,a)
            a1 = epsilon_greedy(qfunc, s1, epsilon)
            key1 = "%d_%s"%(s1,a1)
            qfunc[key] = qfunc[key] + alpha * (r + gamma * qfunc[key1] - qfunc[key])
            s = s1
            a = a1
            count += 1

    plt.plot(x,y,"--",label="sarsa alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    plt.show(True)
    return qfunc;

if __name__ == "__main__":
    read_best()
    sarsa(1000, 0.2, 0.2)

