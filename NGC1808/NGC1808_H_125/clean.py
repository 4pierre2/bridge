#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:33:06 2021

@author: pierre
"""

import glob
import sys
import numpy as np
from astropy.io import fits


sys.path.insert(0, "/home/pierre/Documents/SINFONI_ANALYSIS/v1/")
from sinfobj import sinfobj
from merge_and_clean_fits import shift, make_new_fits, make_new_sinfobj




filenames = glob.glob('/media/pierre/Disque_2/SNR/CALIBRATED_1/*1808*/H_124/*/*.fits')

objs = []

forbidden_ones = ["N_2"]

for filename in filenames:
    for one in forbidden_ones:
        if one in filename:
            filenames.remove(filename)

for filename in filenames:
    hdu = fits.open(filename)
    obj = sinfobj(hdu)
    obj.recenter_on_max()
    objs.append(obj)
    hdu.close()
    
ab, RA, dec, lam = shift(objs)

mean = np.mean(ab, 0)
median = np.median(ab, 0)

obj = make_new_sinfobj(filenames[1], median, RA, dec, lam)

obj.crop(-3,4,-3.8,3.8,1.46,1.8)

make_new_fits("./cleaned.fits", filenames[1], obj.data, header=obj.header)

