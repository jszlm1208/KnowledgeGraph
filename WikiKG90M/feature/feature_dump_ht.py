from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
import os
import argparse
import logging
import numpy as np
import pickle
from tqdm import tqdm
import math
from multiprocessing import Pool

test_num = 500000000
output_path = 'outputs'
data_path = '/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full'

val_hr_path = os.path.join(data_path, "wikikg90m-v2/processed/val_hr.npy")
val_t_candidate_path = os.path.join(data_path, "wikikg90m-v2/processed/val_t_candidate.npy")
test_hr_path = os.path.join(data_path, "wikikg90m-v2/processed/test-dev_hr.npy")
test_t_candidate_path = os.path.join(data_path, "wikikg90m-v2/processed/test_t_candidate.npy")
val_hr = np.load(val_hr_path, mmap_mode="r")
val_t_candidate = np.load(val_t_candidate_path, mmap_mode="r")
test_hr = np.load(test_hr_path, mmap_mode="r")
test_t_candidate = np.load(test_t_candidate_path, mmap_mode="r")

val_output_path = os.path.join(output_path,"valid_feats")
test_output_path = os.path.join(output_path,"test_feats")


# HT
def f(x):
    res = np.zeros_like(x)
    unique, counts = np.unique(x, return_counts=True)
    mapper_dict = {}
    for idx, count in zip(unique, counts):
        mapper_dict[idx] = count

    def mp(entry):
        return mapper_dict[entry]

    mp = np.vectorize(mp)
    return mp(x)


# valid
val_h_sorted_index = np.argsort(val_hr[:, 0], axis=0)
val_h_sorted = val_hr[val_h_sorted_index]
val_h_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(val_h_sorted) + 1)):
    if i == len(val_h_sorted):
        val_h_sorted_index_part.append(tmp)
        break
    if val_h_sorted[i][0] > last_start:
        if last_start != -1:
            val_h_sorted_index_part.append(tmp)
        tmp = []
        last_start = val_h_sorted[i][0]
    tmp.append(i)
val_h_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in val_h_sorted_index_part
]
inputs = [
    val_t_candidate[val_h_sorted_index[arr]] for arr in val_h_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
ht_feat = np.zeros_like(val_t_candidate)
for (arr, mapped) in zip(val_h_sorted_index_arr, mapped_array):
    ht_feat[val_h_sorted_index[arr]] = mapped
np.save("%s/valid_feats/ht_feat.npy" % output_path, ht_feat.astype(np.float32))

# test
test_h_sorted_index = np.argsort(test_hr[:, 0], axis=0)
test_h_sorted = test_hr[test_h_sorted_index]
test_h_sorted_index_part = []
last_start = -1
tmp = []
for i in tqdm(range(len(test_h_sorted) + 1)):
    if i == len(test_h_sorted):
        test_h_sorted_index_part.append(tmp)
        break
    if test_h_sorted[i][0] > last_start:
        if last_start != -1:
            test_h_sorted_index_part.append(tmp)
        tmp = []
        last_start = test_h_sorted[i][0]
    tmp.append(i)
test_h_sorted_index_arr = [
    np.array(
        idx, dtype="int32") for idx in test_h_sorted_index_part
]
inputs = [
    test_t_candidate[test_h_sorted_index[arr]]
    for arr in test_h_sorted_index_arr
]
mapped_array = None
with Pool(20) as p:
    mapped_array = list(tqdm(p.imap(f, inputs), total=len(inputs)))
ht_feat = np.zeros_like(test_t_candidate)
for (arr, mapped) in zip(test_h_sorted_index_arr, mapped_array):
    ht_feat[test_h_sorted_index[arr]] = mapped
np.save("%s/test_feats/ht_feat.npy" % output_path, ht_feat.astype(np.float32))