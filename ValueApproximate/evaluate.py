#!/usr/bin/python
# -*- coding:UTF-8 -*-

import grid_mdp

class Evaler:
    def __init__(self, grid):
        self.grid = grid
        self.best = dict()
        f = open("eval.data")
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            element = line.split(":")
            self.best[element[0]] = float(element[1])

    def eval(self, policy):
        grid = self.grid
        sum1 = 0.0
        for key in self.best:
            keys = key.split("_")
            s = int(keys[0])
            if s in grid.terminal_states:
                continue

            f = grid.start(s)
            a = keys[1]

            error = policy.qfunc(f, a) - self.best[key]
            sum1 += error * error

        return sum1
