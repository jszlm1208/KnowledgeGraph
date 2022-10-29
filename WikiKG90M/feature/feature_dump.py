from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging
import numpy as np
import pickle
from tqdm import tqdm
import math
from multiprocessing import Pool

##dim_r=1315 for v1 data, 1387 for v2 data

def get_rrt_feat(t_candidate, hr,t2r_prob,r2t_prob,dim_r=1387):
    rrt = np.zeros((dim_r, dim_r))
    for i in tqdm(range(dim_r)):
        for t in r2t_prob[i]:
            prob = r2t_prob[i][t]
            for r in t2r_prob[t]:
                prob2 = t2r_prob[t][r]
                rrt[i, r] += prob * prob2
    rrt_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    prob = rrt[r1, r2] * r2t_prob[r2][tail]
                    rrt_feat[i, j] += prob
    return rrt_feat


def get_h2t_t2h_feat(t_candidate, hr,h2t_prob,t2h_prob):
    h2t_t2h_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in h2t_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in h2t_prob:
                continue
            for e in h2t_prob[h]:
                if e not in h2t_prob[tail]:
                    continue
                prob = h2t_prob[h][e] * t2h_prob[e][tail]
                h2t_t2h_feat[i][j] += prob
    return h2t_t2h_feat

def get_t2h_h2t_feat(t_candidate, hr,h2t_prob,t2h_prob):
    t2h_h2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in t2h_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in t2h_prob:
                continue
            for e in t2h_prob[h]:
                if e not in t2h_prob[tail]:
                    continue
                prob = t2h_prob[h][e] * h2t_prob[e][tail]
                t2h_h2t_feat[i][j] += prob
    return t2h_h2t_feat

def get_h2t_h2t_feat(t_candidate, hr,h2t_prob,t2h_prob):
    h2t_h2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in h2t_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in t2h_prob:
                continue
            for e in h2t_prob[h]:
                if e not in t2h_prob[tail]:
                    continue
                prob = h2t_prob[h][e] * h2t_prob[e][tail]
                h2t_h2t_feat[i][j] += prob
    return h2t_h2t_feat

def get_hht_feat(t_candidate, hr,h2t_prob,t2h_prob):
    hh_byt_prob = dict()
    for h in tqdm(h2t_prob):
        if len(h2t_prob[h]) > 10:
            continue
        for t in h2t_prob[h]:
            prob = h2t_prob[h][t]
            if len(t2h_prob[t]) > 10:
                continue
            for h2 in t2h_prob[t]:
                prob2 = t2h_prob[t][h2]
                if h not in hh_byt_prob:
                    hh_byt_prob[h] = dict()
                if h2 not in hh_byt_prob[h]:
                    hh_byt_prob[h][h2] = prob * prob2
                else:
                    hh_byt_prob[h][h2] += prob * prob2
    hht_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h1 = hr[i, 0]
        if h1 not in hh_byt_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2h_prob:
                for h2 in t2h_prob[tail]:
                    if h2 not in hh_byt_prob[h1]:
                        continue
                    prob = hh_byt_prob[h1][h2] * h2t_prob[h2][tail]
                    hht_feat[i, j] += prob
    return hht_feat

def get_r2t_h2r_feat(t_candidate, hr, h2r_prob,r2t_prob,r2h_prob,dim_r=1387):
    r2t_h2r = np.zeros((dim_r, dim_r), dtype=np.float16)
    for i in tqdm(range(dim_r)):
        for t in r2t_prob[i]:
            prob = r2t_prob[i][t]
            if t not in h2r_prob:
                continue
            for r in h2r_prob[t]:
                prob2 = h2r_prob[t][r]
                r2t_h2r[i, r] += prob * prob2
    r2t_h2r_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in h2r_prob:
                for r2 in h2r_prob[tail]:
                    prob = r2t_h2r[r1, r2] * r2h_prob[r2][tail]
                    r2t_h2r_feat[i, j] += prob
    return r2t_h2r_feat

def get_r2t_feat(t_candidate, hr, r2t_prob):
    r2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            t = t_candidate[i, j]
            if t in r2t_prob[r]:
                r2t_feat[i, j] = r2t_prob[r][t]
    return r2t_feat

def get_rrh_feat(t_candidate, hr, h2r_prob,r2h_prob,t2r_prob,r2t_prob,dim_r=1387):
    rrh = np.zeros((dim_r, dim_r))
    for i in tqdm(range(dim_r)):
        for h in r2h_prob[i]:
            prob = r2h_prob[i][h]
            for r in h2r_prob[h]:
                prob2 = h2r_prob[h][r]
                rrh[i, r] += prob * prob2
    rrh_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    if tail not in r2t_prob[r2]:
                        print(r1, r2, tail)
                        exit()
                    prob = rrh[r1, r2] * r2t_prob[r2][tail]
                    rrh_feat[i, j] += prob
    return rrh_feat
