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
    return sum1

def mc(num_iter1, epsilon):
    x = []
    y = []
    n = dict()
    qfunc = dict()
    for s in states:
        for a in actions:
            qfunc["%d_%s"%(s,a)] = 0.0
            n["%d_%s"%(s,a)] = 0.001

    for iter1 in xrange(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))

        #generate example
        s_sample = []
        a_sample = []
        r_sample = []

        s = states[int(random.random() * len(states))]
        t = False
        count = 0
        while False == t and count < 100:
            a = epsilon_greedy(qfunc, s, epsilon)
            t, s1, r = grid.transform(s, a)
            s_sample.append(s)
            r_sample.append(r)
            a_sample.append(a)
            s = s1
            count += 1

        g = 0.0
        for i in xrange(len(r_sample)-1, -1, -1):
            g *= gamma
            g += r_sample[i]

        for i in xrange(len(s_sample)):
            key = "%d_%s"%(s_sample[i], a_sample[i])
            qfunc[key] = (qfunc[key] * n[key] + g) / (n[key] + 1)
            n[key] += 1.0

            g -= r_sample[i]
            g /= gamma

    plt.plot(x,y,"-",label="mc epsilon=%2.1f"%(epsilon))
    plt.show(True)
    return qfunc

if __name__ == "__main__":
    read_best()
    qfunc = mc(1000, 0.2)
    print(qfunc)


