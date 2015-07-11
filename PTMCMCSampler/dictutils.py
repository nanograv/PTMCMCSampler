# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict
import numpy as np


def make_ordered_dict(keys, values):
    x = OrderedDict({})
    for key, value in zip(keys,values):
        x[key] = value
    return x

def get_dict_subset(pdict, ind):
    
    keys, values = [], []
    for ii in ind:
        key = [key for key in pdict if pdict[key]==ii][0]
        value = pdict[key]
        
        keys.append(key)
        values.append(value)
    
    return make_ordered_dict(keys, values)

def match_dict_values(pdict1, pdict2):
    
    return np.array([pdict1[key] for key in pdict1 if key in pdict2])
