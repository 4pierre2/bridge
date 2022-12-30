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



filenames = glob.glob('/media/pierre/Disque_2/SNR/CALIBRATED_1/*M77*/K_50/*/*.fits')

objs = []

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

obj.crop(-0.9,0.9,-0.9,0.9,1.46,2.455)

make_new_fits("./cleaned.fits", filenames[1], obj.data)

