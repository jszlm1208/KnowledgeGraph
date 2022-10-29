import numpy as np
import pickle
from tqdm import tqdm


def h2r(train_hrt):
    data = dict()
    for h, r, t in tqdm(train_hrt):
        if h not in data:
            data[h] = dict()
        if not r in data[h]:
            data[h][r] = 1
        else:
            data[h][r] += 1

    del train_hrt

    for h in data:
        h_sum = sum(data[h].values())
        for r in data[h]:
            data[h][r] /= h_sum

    return data

def h2t(train_hrt):
    data = dict()

    for h, r, t in tqdm(train_hrt):
        if h not in data:
            data[h] = dict()
        if not t in data[h]:
            data[h][t] = 1
        else:
            data[h][t] += 1

    del train_hrt

    for h in tqdm(data):
        h_sum = sum(data[h].values())
        for t in data[h]:
            data[h][t] /= h_sum
    return data

def r2h(train_hrt):
    data = [dict() for _ in range(1387)]

    for h, r, t in tqdm(train_hrt):
        if not h in data[r]:
            data[r][h] = 1
        else:
            data[r][h] += 1

    for r in range(1387):
        r_sum = sum(data[r].values())
        for h in data[r]:
            data[r][h] /= r_sum
    return data

def r2t(train_hrt):
    data = [dict() for _ in range(1387)]

    for h, r, t in tqdm(train_hrt):
        if not t in data[r]:
            data[r][t] = 1
        else:
            data[r][t] += 1

    for r in range(1387):
        r_sum = sum(data[r].values())
        for t in data[r]:
            data[r][t] /= r_sum
    return data

def t2h(train_hrt):
    data = dict()
    for h, r, t in tqdm(train_hrt):
        if t not in data:
            data[t] = dict()
        if not h in data[t]:
            data[t][h] = 1.
        else:
            data[t][h] += 1.


    del train_hrt

    for t in tqdm(data):
        t_sum = sum(data[t].values())
        for h in data[t]:
            data[t][h] /= t_sum
    return data

def t2r(train_hrt):
    data = dict()

    for h, r, t in tqdm(train_hrt):
        if t not in data:
            data[t] = dict()
        if not r in data[t]:
            data[t][r] = 1
        else:
            data[t][r] += 1

    for t in data:
        t_sum = sum(data[t].values())
        for r in data[t]:
            data[t][r] /= t_sum
    return data








   