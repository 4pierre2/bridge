#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:06:07 2022

@author: pierre
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage
from scipy.optimize import curve_fit
import glob
from tqdm import tqdm

from astropy.io import fits
from copy import deepcopy

import sys
sys.path.append('../')

from bridge_model import stellar_bridge, G, pc, M_sol
import bridge_model as br
from sinfobj import sinfobj, extinct


#%%
  
# Params


dist = 12.8e6


rebin_fac_cont = 50
rebin_fac_CO = 1

vs_cont = np.linspace(-2, 2, 2)*1e3
stds_cont = np.linspace(-1, 1, 2)*1e3
vs_CO = np.arange(-400, 401, 20)*1e3
stds_CO = np.arange(0, 201, 20)*1e3
aks = np.arange(0, 1, 0.05)

n_pix_1 = 25
n_pix_2 = 25

#%%

# Load SINFO data
print('Loading SINFO data')

filenamej = "/home/pierre//Documents/CLEANED_1/NGC1808_J_125/cleaned.fits"     
filenameh = "/home/pierre//Documents/CLEANED_1/NGC1808_H_125/cleaned.fits"     
filenamek = "/home/pierre//Documents/CLEANED_1/NGC1808_K_125/cleaned.fits"     
      
hduj = fits.open(filenamej)
objj = sinfobj(hduj, z=0)    
lamj = objj.lam*(2.292/2.302)
dlamj = lamj[1]-lamj[0]      
cubej = objj.data[:, 6:55, 8:57]/dlamj
shapej = np.shape(cubej)
imj = objj.get_im(filt=np.loadtxt(objj.filter_names[0]).T)
magj = -2.5*np.log10(imj/3.129e-13/0.162)
hduh = fits.open(filenameh)
objh = sinfobj(hduh, z=0)       
lamh = objh.lam*(2.292/2.302)
dlamh = lamh[1]-lamh[0]   
cubeh = objh.data[:, 6:55, 8:57]/dlamh
shapeh = np.shape(cubeh)
imh = objh.get_im(filt=np.loadtxt(objh.filter_names[0]).T)
magh = -2.5*np.log10(imh/1.133e-13/0.251)
hduk = fits.open(filenamek)
objk = sinfobj(hduk, z=0)   
lamk = objk.lam*(2.292/2.302)
dlamk = lamk[1]-lamk[0]       
cubek = objk.data[:, 6:55, 8:57]/dlamk
shapek = np.shape(cubek)
imk = objk.get_im(filt=np.loadtxt(objk.filter_names[0]).T)
magk = -2.5*np.log10(imk/4.283e-14/0.262)


#%%

# Load SSPs
print("Loading SSPs")

masses = []
ssps_t = []
ages = []

path = '/home/pierre/Documents/SPISEA/SSPS_1/Z0_1A/'


ages_masses = []
filenames = glob.glob(path+'/*masses.npy')
for filename in filenames:
    age = float(filename.split('/')[-1].split('all_')[0])
    ages_masses.append(age)
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    mas = np.load(filename)
    ma = 0
    for m in mas:
        m = np.array(m)
        where = m[:,0] > 3
        ma += np.sum(m[:,0][where]*m[:,1][where])
    masses.append(ma)
    np.load = np_load_old
    
masses = [x for _, x in sorted(zip(ages_masses, masses))]

filenames = glob.glob(path+'/*.txt')
for filename in filenames:
    if 'spec.txt' in filename:
        age = float(filename.split('/')[-1].split('_spec.tx')[0])
        ages.append(age)
        ssps_t.append(np.loadtxt(filename))
    elif 'lam.txt' in filename:
        lam_ssp = np.loadtxt(filename)

ages, ssps_t = (list(t) for t in zip(*sorted(zip(ages, ssps_t))))
ssps = deepcopy(ssps_t)

# Flux conversions

for j in range(len(ssps)):
    ssps[j] /= (dist/10)**2 # Setting at object's distance, i.e. some Mpc
    ssps[j] /= 10**6*M_sol # Converting to W.m-2.um-1 per kg
    
    
#%%

# Make continuum data
print("Making continuum data")

lam = np.concatenate([lamj, lamh, lamk])

lam_rebinned_j_cont  = br.rebin_1d(lamj, int(len(lamj)/rebin_fac_cont))
lam_rebinned_h_cont  = br.rebin_1d(lamh, int(len(lamh)/rebin_fac_cont))
lam_rebinned_k_cont  = br.rebin_1d(lamk, int(len(lamk)/rebin_fac_cont))
lam_rebinned_cont  = np.concatenate([lam_rebinned_j_cont, lam_rebinned_h_cont, lam_rebinned_k_cont])

l0j = np.argmin(abs(lam_ssp-np.min(lamj)))
l1j = np.argmin(abs(lam_ssp-np.max(lamj)))
lamj_ssp = lam_ssp[l0j:l1j]
l0h = np.argmin(abs(lam_ssp-np.min(lamh)))
l1h = np.argmin(abs(lam_ssp-np.max(lamh)))
lamh_ssp = lam_ssp[l0h:l1h]
l0k = np.argmin(abs(lam_ssp-np.min(lamk)))
l1k = np.argmin(abs(lam_ssp-np.max(lamk)))
lamk_ssp = lam_ssp[l0k:l1k]


ssps_cropped_j = []
ssps_cropped_h = []
ssps_cropped_k = []
for ssp in ssps:
    sspj = ssp[l0j:l1j]
    ssph = ssp[l0h:l1h]
    sspk = ssp[l0k:l1k]
    ssps_cropped_j.append(sspj)
    ssps_cropped_h.append(ssph)
    ssps_cropped_k.append(sspk)
    
ssps_rebinned_cont = []
ssps_rebinned_j_cont = []
ssps_rebinned_h_cont = []
ssps_rebinned_k_cont = []
lam_ssp_rebinned_j_cont  = br.rebin_1d(lamj_ssp, int(len(lamj)/rebin_fac_cont))
lam_ssp_rebinned_h_cont  = br.rebin_1d(lamh_ssp, int(len(lamh)/rebin_fac_cont))
lam_ssp_rebinned_k_cont  = br.rebin_1d(lamk_ssp, int(len(lamk)/rebin_fac_cont))
lam_ssp_rebinned_cont  = np.concatenate([lam_ssp_rebinned_j_cont, lam_ssp_rebinned_h_cont, lam_ssp_rebinned_k_cont])

for n in range(len(ssps_cropped_j)):
    ssp_cropped_j = ssps_cropped_j[n]
    ssp_cropped_h = ssps_cropped_h[n]
    ssp_cropped_k = ssps_cropped_k[n]
    ssp_rebinned_j = br.rebin_1d(ssp_cropped_j, len(lam_ssp_rebinned_j_cont))#*len(ssp_cropped_j)/len(lam_ssp_rebinned_j_cont)
    ssp_rebinned_h = br.rebin_1d(ssp_cropped_h, len(lam_ssp_rebinned_h_cont))#*len(ssp_cropped_h)/len(lam_ssp_rebinned_h_cont)
    ssp_rebinned_k = br.rebin_1d(ssp_cropped_k, len(lam_ssp_rebinned_k_cont))#*len(ssp_cropped_k)/len(lam_ssp_rebinned_k_cont)
    ssp_rebinned = np.concatenate([ssp_rebinned_j, ssp_rebinned_h, ssp_rebinned_k])
    ssps_rebinned_j_cont.append(ssp_rebinned_j)
    ssps_rebinned_h_cont.append(ssp_rebinned_h)
    ssps_rebinned_k_cont.append(ssp_rebinned_k)
    ssps_rebinned_cont.append(ssp_rebinned)
    
lam0_cont = np.mean(lam_ssp_rebinned_cont)
dlam_cont = np.mean(abs(lam_ssp_rebinned_cont[1:]-lam_ssp_rebinned_cont[:-1]))

mega_ssps_cont = []
for n in range(len(ssps_rebinned_cont)): 
    ssp_cont = ssps_rebinned_cont[n]
    mega_ssp = np.ones((len(stds_cont), len(vs_cont), len(ssp_cont)))
    for i in range(len(stds_cont)):
        # std = (stds_cont[i]+stds_cont[i+1])/2
        std = stds_cont[i]
        ssp_smoothed =  scipy.ndimage.gaussian_filter(ssp_cont, lam0_cont/dlam_cont*std/299792458)
        for j in range(len(vs_cont)):
            # v = (vs_cont[j]+vs_cont[j+1])/2
            v = vs_cont[j]
            ssp_moved = br.redshift(lam_ssp_rebinned_cont, ssp_smoothed, v, velocity=True)
            mega_ssp[i, j] = ssp_moved
    mega_ssps_cont.append(np.array(mega_ssp))
    

extinctions_cont = []
for k in range(len(aks)):
    ak = -aks[k]
    extinctions_cont.append(extinct(lam_ssp_rebinned_cont, ak))

#%%

# Make CO data
print("Making CO data")

lam = np.concatenate([lamj, lamh, lamk])

lam_rebinned_k_CO  = br.rebin_1d(lamk, int(len(lamk)/rebin_fac_CO))
lam_rebinned_CO  = np.concatenate([lam_rebinned_k_CO])

l0k = np.argmin(abs(lam_ssp-2.25))
l1k = np.argmin(abs(lam_ssp-2.4))
l0k_cube = np.argmin(abs(lamk-2.25))
l1k_cube = np.argmin(abs(lamk-2.4))
lamk_ssp = lam_ssp[l0k:l1k]

ssps_cropped_k = []
for ssp in ssps:
    sspk = ssp[l0k:l1k]
    ssps_cropped_k.append(sspk)
    
ssps_rebinned_CO = []
ssps_rebinned_k_CO = []
lam_ssp_rebinned_k_CO  = br.rebin_1d(lamk_ssp, int(len(lamk[l0k_cube:l1k_cube])/rebin_fac_CO))
lam_ssp_rebinned_CO  = np.concatenate([lam_ssp_rebinned_k_CO])

for n in range(len(ssps_cropped_j)):    
    ssp_cropped_k = ssps_cropped_k[n]
    ssp_rebinned_k = br.rebin_1d(ssp_cropped_k, len(lam_ssp_rebinned_k_CO))#*len(ssp_cropped_k)/len(lam_ssp_rebinned_k_CO)
    ssp_rebinned = np.concatenate([ssp_rebinned_k])
    ssps_rebinned_k_CO.append(ssp_rebinned_k)
    ssps_rebinned_CO.append(ssp_rebinned)


lam0_CO = np.mean(lam_ssp_rebinned_CO)
dlam_CO = np.mean(abs(lam_ssp_rebinned_CO[1:]-lam_ssp_rebinned_CO[:-1]))

mega_ssps_CO = []
for n in range(len(ssps_rebinned_CO)): 
    ssp_CO = ssps_rebinned_CO[n]
    mega_ssp_CO = np.ones((len(stds_CO), len(vs_CO), len(ssp_CO)))
    for i in range(len(stds_CO)):
        # std = (stds_CO[i]+stds_CO[i+1])/2
        std = stds_CO[i]
        ssp_smoothed =  scipy.ndimage.gaussian_filter(ssp_CO, lam0_CO/dlam_CO*std/299792458)
        for j in range(len(vs_CO)):
            # v = (vs_CO[j]+vs_CO[j+1])/2
            v = vs_CO[j]
            ssp_moved = br.redshift(lam_ssp_rebinned_k_CO, ssp_smoothed, v, velocity=True)
            mega_ssp_CO[i, j] = ssp_moved
    mega_ssps_CO.append(np.array(mega_ssp_CO))

extinctions_CO = []
for k in range(len(aks)):
    ak = -aks[k]
    extinctions_CO.append(extinct(lam_ssp_rebinned_CO, ak))
            
#%%
    
print("Rebinning cube to continuum")
cube_cont_j_1 = br.rebin_nd(cubej, (int(len(lamj)/rebin_fac_cont), n_pix_1, n_pix_1))*shapej[1]*shapej[2]/n_pix_1**2
cube_cont_h_1 = br.rebin_nd(cubeh, (int(len(lamh)/rebin_fac_cont), n_pix_1, n_pix_1))*shapeh[1]*shapeh[2]/n_pix_1**2
cube_cont_k_1 = br.rebin_nd(cubek, (int(len(lamk)/rebin_fac_cont), n_pix_1, n_pix_1))*shapek[1]*shapek[2]/n_pix_1**2
cube_cont = np.concatenate([cube_cont_j_1, cube_cont_h_1, cube_cont_k_1])


print("Rebinning cube to CO")

cube_CO = br.rebin_nd(cubek[l0k_cube:l1k_cube], (int(len(lamk[l0k_cube:l1k_cube])/rebin_fac_CO), n_pix_1, n_pix_1))*shapek[1]*shapek[2]/n_pix_1**2
cube_CO_EW = cube_CO/br.get_raw_cont(cube_CO, n=30)
cube_CO_sub = cube_CO-br.get_raw_cont(cube_CO, n=30)

#%%
print("Getting NSC and disk spectra")

mask_NSC_cont = np.zeros(np.shape(cube_cont), dtype=bool)
mask_NSC_CO = np.zeros(np.shape(cube_CO), dtype=bool)
mask_disk_cont = np.zeros(np.shape(cube_cont), dtype=bool)
mask_disk_CO = np.zeros(np.shape(cube_CO), dtype=bool)

mask_NSC_cont[:, 10:15, 10:15] = True
mask_NSC_CO[:, 10:15, 10:15] = True
mask_disk_cont[:, 8:17, 8:17] = True
mask_disk_CO[:, 8:17, 8:17] = True
mask_disk_cont[mask_NSC_cont] = False
mask_disk_CO[mask_NSC_CO] = False


cube_NSC_cont = cube_cont*mask_NSC_cont
cube_NSC_CO = cube_CO*mask_NSC_CO

cube_disk_cont = cube_cont*mask_disk_cont
cube_disk_cont_full = cube_cont*(1-mask_NSC_cont)
cube_disk_CO = cube_CO*mask_disk_CO

cont_disk = np.sum(cube_disk_cont, (1,2))*np.sum(mask_NSC_cont[0])/np.sum(mask_disk_cont[0])
cont_disk_full = np.sum(cube_disk_cont_full, (1,2))
cont_NSC = np.sum(cube_NSC_cont, (1,2))-cont_disk

CO_disk = np.mean(cube_disk_CO, (1,2))*np.sum(mask_NSC_CO[0])/np.sum(mask_disk_CO[0])
CO_NSC = np.mean(cube_NSC_CO, (1,2))-CO_disk

CO_NSC_EW = CO_NSC/br.get_raw_cont(CO_NSC[:, None, None])[:,0,0]
CO_disk_EW = CO_disk/br.get_raw_cont(CO_disk[:, None, None])[:,0,0]


obs_NSC = np.concatenate([cont_NSC/np.mean(cont_NSC), CO_NSC_EW])
obs_disk = np.concatenate([cont_disk/np.mean(cont_disk), CO_disk_EW])

#%%
print("Fitting NSC ages and extinction")

k2min = 1e100
k2s_cont = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
k2s_CO = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
k2s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
ratios = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
a1s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])), dtype=int)
a2s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])), dtype=int)
akss = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))


i = 0
for i_ak in tqdm(range(len(aks))):
    ak = aks[i_ak]
    ext_cont = extinct(lam_ssp_rebinned_cont, -ak)
    j = 0
    for i_a1 in range(len(ages))[:150:10]:
        a1 = ages[i_a1]
        ssp_cont_1 = ssps_rebinned_cont[i_a1]*ext_cont
        ssp_CO_1 = ssps_rebinned_CO[i_a1]
        k = 0
        for i_a2 in range(len(ages))[300::10]:
            a2 = ages[i_a2]
            ssp_cont_2 = ssps_rebinned_cont[i_a2]*ext_cont
            ssp_CO_2 = ssps_rebinned_CO[i_a2]
            
            
            def f_for_fit(_, ratio):
                cont =  ratio*ssp_cont_1+(1-ratio)*ssp_cont_2
                CO =  ratio*ssp_CO_1+(1-ratio)*ssp_CO_2
                CO_EW = CO/br.get_raw_cont(CO[:, None, None])[:, 0, 0]
                return np.concatenate([cont/np.mean(cont), CO_EW])
            
            
            p, c = curve_fit(f_for_fit, None, obs_NSC, p0 = [0.05], bounds = (0, 0.1))
            mod = f_for_fit(None, *p)
            a1s[i, j, k] = i_a1
            a2s[i, j, k] = i_a2
            akss[i, j, k] = ak
            ratios[i, j, k] = p
            k2 = np.sum((mod-obs_NSC)**2)
            k2s[i, j, k] = k2
            if k2 < k2min:
                k2min = k2
                best_mod = mod
                CO =  p*ssp_CO_1+(1-p)*ssp_CO_2
                best_CO_sub_NSC = CO-br.get_raw_cont(CO[:, None, None])[:, 0, 0]   

            k += 1
        j += 1
    i += 1
       
argmin = np.unravel_index(np.argmin(k2s), np.shape(k2s))

print("Shape:", np.shape(k2s))
print("Argmin: ", argmin)
best_i_a1_NSC = a1s[argmin]
best_i_a2_NSC = a2s[argmin]
best_a1_NSC = ages[a1s[argmin]]
best_a2_NSC = ages[a2s[argmin]]
best_ratio_NSC = ratios[argmin]
best_ak_NSC = akss[argmin]

print("Young age:", np.log10(best_a1_NSC))
print("Old age:", np.log10(best_a2_NSC))
print("Young age ratio: ", best_ratio_NSC)
print("A_k: ", best_ak_NSC)

best_cont = best_mod[:len(cont_NSC)]
best_CO = best_mod[len(cont_NSC):]
best_CO_NSC = best_mod[len(cont_NSC):]

print("Required log(M_sol): ", np.log10(np.mean(cont_NSC)/(np.mean(ssps_rebinned_cont[a1s[argmin]]*best_ratio_NSC+ssps_rebinned_cont[a2s[argmin]]*(1-best_ratio_NSC)))/M_sol))
print("Flux ratio (young/old)", np.mean(ssps_rebinned_cont[a1s[argmin]]*best_ratio_NSC+ssps_rebinned_cont[a2s[argmin]]*(1-best_ratio_NSC)))
plt.figure()
plt.plot(lam_ssp_rebinned_CO, best_CO, label="Model")
plt.plot(lam_ssp_rebinned_CO, CO_NSC_EW, label="Obs NSC")

plt.figure()
plt.plot(lam_ssp_rebinned_cont, best_cont, label="Model")
plt.plot(lam_ssp_rebinned_cont, cont_NSC/np.mean(cont_NSC), label="Obs NSC")

#%%
print("Fitting disk ages and extinction")

k2min = 1e100
k2s_cont = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
k2s_CO = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
k2s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
ratios = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))
a1s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])), dtype=int)
a2s = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])), dtype=int)
akss = np.zeros((len(aks), len(ages[:150:10]), len(ages[300::10])))


i = 0
for i_ak in tqdm(range(len(aks))):
    ak = aks[i_ak]
    ext_cont = extinct(lam_ssp_rebinned_cont, -ak)
    j = 0
    for i_a1 in range(len(ages))[:150:10]:
        a1 = ages[i_a1]
        ssp_cont_1 = ssps_rebinned_cont[i_a1]*ext_cont
        ssp_CO_1 = ssps_rebinned_CO[i_a1]
        k = 0
        for i_a2 in range(len(ages))[300::10]:
            a2 = ages[i_a2]
            ssp_cont_2 = ssps_rebinned_cont[i_a2]*ext_cont
            ssp_CO_2 = ssps_rebinned_CO[i_a2]
            
            
            def f_for_fit(_, ratio):
                cont =  ratio*ssp_cont_1+(1-ratio)*ssp_cont_2
                CO =  ratio*ssp_CO_1+(1-ratio)*ssp_CO_2
                CO_EW = CO/br.get_raw_cont(CO[:, None, None])[:, 0, 0]
                return np.concatenate([cont/np.mean(cont), CO_EW])
            
            
            p, c = curve_fit(f_for_fit, None, obs_disk, p0 = [0.05], bounds = (0, 1))
            mod = f_for_fit(None, *p)
            a1s[i, j, k] = i_a1
            a2s[i, j, k] = i_a2
            akss[i, j, k] = ak
            ratios[i, j, k] = p
            k2 = np.sum((mod-obs_disk)**2)
            k2s[i, j, k] = k2
            if k2 < k2min:
                k2min = k2
                best_mod = mod
                CO =  p*ssp_CO_1+(1-p)*ssp_CO_2
                best_CO_sub_disk = CO-br.get_raw_cont(CO[:, None, None])[:, 0, 0]   
            k += 1
        j += 1
    i += 1
       
argmin = np.unravel_index(np.argmin(k2s), np.shape(k2s))

print("Shape:", np.shape(k2s))
print("Argmin: ", argmin)
best_i_a1_disk = a1s[argmin]
best_i_a2_disk = a2s[argmin]
best_a1_disk = ages[a1s[argmin]]
best_a2_disk = ages[a2s[argmin]]
best_ratio_disk = ratios[argmin]
best_ak_disk = akss[argmin]

print("Young age:", np.log10(best_a1_disk))
print("Old age:", np.log10(best_a2_disk))
print("Young age ratio: ", best_ratio_disk)
print("A_k: ", best_ak_disk)

best_cont = best_mod[:len(cont_disk)]
best_CO = best_mod[len(cont_disk):]
best_CO_disk = best_mod[len(cont_disk):]

print("Required log(M_sol): ", np.log10(np.mean(cont_disk_full)/(np.mean(ssps_rebinned_cont[a1s[argmin]]*best_ratio_disk+ssps_rebinned_cont[a2s[argmin]]*(1-best_ratio_disk)))/M_sol))
print("Flux ratio (young/old)", np.mean(ssps_rebinned_cont[a1s[argmin]])/np.mean(ssps_rebinned_cont[a2s[argmin]]))
plt.figure()
plt.plot(lam_ssp_rebinned_CO, best_CO, label="Model")
plt.plot(lam_ssp_rebinned_CO, CO_disk_EW, label="Obs disk")

plt.figure()
plt.plot(lam_ssp_rebinned_cont, best_cont, label="Model")
plt.plot(lam_ssp_rebinned_cont, cont_disk/np.mean(cont_disk), label="Obs disk")


#%%


#%%
mesh = br.compute_mesh(n_pix_1, np.shape(objj.data)[1]/n_pix_1*objj.pix_scale*62*pc)
     

def f_for_ifus(_, M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio):
                    plum = [br.plummer_mesh_density,
                            br.plummer_mesh_cyl_vel,
                            br.zero,
                            'Null',
                            br.zero,
                            br.plummer_mesh_velocity_dispersion,
                            'Uniform',
                            False,
                            [M_plum*M_sol, R_plum*pc],
                            0, 0]
    
                    disk = [br.miyamoto_nagai_mesh_density,
                            br.miyamoto_nagai_mesh_cyl_vel,
                            br.zero,
                            'Null',
                            br.miyamoto_nagai_mesh_cyl_vel_disp,
                            br.zero,
                            'Null',
                            True,
                            [M_disk*M_sol, R_disk*pc, R_disk/abs(aspect_ratio)*pc],
                            i, PA]
    
    
                    objb = stellar_bridge(mesh, [plum, disk])
                    
                    
                    ext_cont_1 = extinct(lam_ssp_rebinned_cont, -best_ak_NSC)
                    ext_cont_2 = extinct(lam_ssp_rebinned_cont, -best_ak_disk)
                    mega_ssp_cont_plum = ext_cont_1[None, None, :]*(mega_ssps_cont[best_i_a1_NSC]*young_ratio_plum+mega_ssps_cont[best_i_a2_NSC]*(1-young_ratio_plum))
                    mega_ssp_cont_disk = ext_cont_2[None, None, :]*(mega_ssps_cont[best_i_a1_disk]*young_ratio_disk+mega_ssps_cont[best_i_a2_disk]*(1-young_ratio_disk))
                    ifu_cont = objb.compute_ifu([mega_ssp_cont_plum, mega_ssp_cont_disk], stds_cont, vs_cont)
                    objb = stellar_bridge(mesh, [plum, disk])
                    mega_ssp_CO_plum = (mega_ssps_CO[best_i_a1_NSC]*young_ratio_plum+mega_ssps_CO[best_i_a2_NSC]*(1-young_ratio_plum))
                    mega_ssp_CO_disk = (mega_ssps_CO[best_i_a1_disk]*young_ratio_disk+mega_ssps_CO[best_i_a2_disk]*(1-young_ratio_disk))
                    ifu_CO = objb.compute_ifu([mega_ssp_CO_plum, mega_ssp_CO_plum], stds_CO, vs_CO)

                    ifu_cont_CO = br.get_raw_cont(ifu_CO)
                    ifu_CO /= ifu_cont_CO

                    return ifu_cont, ifu_CO, objb
                

#%%

from scipy.ndimage import shift, gaussian_filter

def shift_and_smooth(spec, shifty, std, a):
    smoothed = gaussian_filter(a*spec, np.max([std, 0.01]))
    shifted = shift(smoothed, shifty, order=1, mode='nearest')
    return shifted

def shift_and_smooth_EW(spec, shifty, std, a):
    smoothed = gaussian_filter(spec, np.max([std, 0.01]))
    shifted = shift(smoothed, shifty, order=1, mode='nearest')
    return 1+a*(shifted-1)

def fit_losv_losvd(spec, template):
    k2min = 1e100
    for shifty in np.arange(-10, 10, 1):
        k2 = np.sum((spec-shift(template, shifty, order=1, mode='nearest'))**2)
        if k2 < k2min:
            k2min = k2
            shift0 = shifty
    
    k2min = 1e100
    for std in np.arange(1, 2, 0.05):
        k2 = np.sum((spec-gaussian_filter(template, std))**2)
        if k2 < k2min:
            k2min = k2
            std0 = std
    p, c = curve_fit(shift_and_smooth, template, spec, p0 = [shift0, std0, np.mean(spec)/np.mean(template)])
    return p, c

def fit_losv_losvd_EW(spec, template, verbose=False):
    k2min = 1e100
    for shifty in np.arange(-10, 10, 0.1):
        k2 = np.sum((spec-shift(template, shifty, order=1, mode='nearest'))**2)
        if k2 < k2min:
            k2min = k2
            shift0 = shifty
    
    k2min = 1e100
    for std in np.arange(1, 2, 0.05):
        k2 = np.sum((spec-gaussian_filter(template, std))**2)
        if k2 < k2min:
            k2min = k2
            std0 = std
    if verbose:
        print(shift0, std0)
    try:
        p, c = curve_fit(shift_and_smooth_EW, template, spec, p0 = [shift0, std0, 1.1], maxfev=10000)
    except:
        p = p0
        c = p*0
    if verbose:
        plt.plot(shift_and_smooth_EW(template, *p))
        plt.plot(spec)
    return p, c

def fit_losv_losvd_ifu(ifu, ifu0):
    losv = np.zeros(np.shape(ifu[0]))
    losvd = np.zeros(np.shape(ifu[0]))
    losv_err = np.zeros(np.shape(ifu[0]))
    losvd_err = np.zeros(np.shape(ifu[0]))
    for i in range(len(losv)):
        for j in range(len(losv[i])):
            spec = ifu[:, i, j]
            template = ifu0[:, i, j]
            p, c = fit_losv_losvd(spec, template)
            losv[i, j], losvd[i, j], _ = p
            losv_err[i, j], losvd_err[i, j], _err = np.sqrt(np.diag(c))
    return losv, np.max([losvd, 0.01*np.ones(np.shape(losvd))], 0), _, losv_err, losvd_err, _err
            
def fit_losv_losvd_ifu_template_EW(ifu, template):
    losv = np.zeros(np.shape(ifu[0]))
    losvd = np.zeros(np.shape(ifu[0]))
    losv_err = np.zeros(np.shape(ifu[0]))
    losvd_err = np.zeros(np.shape(ifu[0]))
    for i in range(len(losv)):
        for j in range(len(losv[i])):
            spec = ifu[:, i, j]
            p, c = fit_losv_losvd_EW(spec, template)
            losv[i, j], losvd[i, j], _ = p
            losv_err[i, j], losvd_err[i, j], _ = np.sqrt(np.diag(c))
    return losv, np.max([losvd, 0.01*np.ones(np.shape(losvd))], 0), losv_err, losvd_err    
        
def fit_losv_losvd_ifu_template(ifu, template):
    losv = np.zeros(np.shape(ifu[0]))
    losvd = np.zeros(np.shape(ifu[0]))
    losv_err = np.zeros(np.shape(ifu[0]))
    losvd_err = np.zeros(np.shape(ifu[0]))
    for i in range(len(losv)):
        for j in range(len(losv[i])):
            spec = ifu[:, i, j]
            p, c = fit_losv_losvd(spec, template)
            losv[i, j], losvd[i, j], _ = p
            losv_err[i, j], losvd_err[i, j], _ = np.sqrt(np.diag(c))
    return losv, np.max([losvd, 0.01*np.ones(np.shape(losvd))], 0), losv_err, losvd_err
      
t0 = time.time()   
losvs = []
losvds = []
for i in tqdm(np.random.randint(len(ssps_rebinned_CO), size=10)): 
    CO = ssps_rebinned_CO[i]
    template = gaussian_filter(CO-br.get_raw_cont(CO[:, None, None])[:, 0, 0],1)
    # losv, losvd, losv_err, losvd_err = fit_losv_losvd_ifu_template(cube_CO_sub/np.mean(cube_CO_sub), best_CO_sub_disk/np.mean(best_CO_sub_disk))
    losv, losvd, losv_err, losvd_err = fit_losv_losvd_ifu_template(cube_CO_sub/np.mean(cube_CO_sub), template)
    losvs.append(losv)
    losvds.append(losvd)
losv = np.median(losvs, 0)
losvd = np.median(losvds, 0)
t1 = time.time()
print(t1-t0)
v_fac = 3e8*(lam_ssp_rebinned_CO[50]-lam_ssp_rebinned_CO[49])/lam_ssp_rebinned_CO[49]
losvd *= v_fac
losv = scipy.ndimage.median_filter(losv, (3,3))
losvd = scipy.ndimage.median_filter(losvd, (5,5))
losv *= v_fac
losv -= np.median(losv)
# losv += 4e4

losvd_err *= v_fac
losv_err *= v_fac


losvd[losvd<3e4]=3e4
# losv = np.loadtxt('/home/pierre/Documents/BRIDGE/V5/NGC1808/losv.txt')
# losvd = np.loadtxt('/home/pierre/Documents/BRIDGE/V5/NGC1808/losvd.txt')

#%%

M_plums = 10**(np.arange(8.4, 8.41, 0.1))
young_ratio_plums = np.arange(0.16, .161, 0.01)
R_plums = np.arange(50, 51, 5)
M_disks = 10**(np.arange(10.6, 10.61, 0.2))
young_ratio_disks = np.arange(0.08, 0.171, 0.02)
R_disks = np.arange(425, 435, 25)
iss = np.arange(56*np.pi/180, 57*np.pi/180, 2*np.pi/180)
PAs = np.arange(235.8*np.pi/180, 235.9*np.pi/180, np.pi/180)
aspect_ratios = np.arange(1., 1.01, 0.4)

k2s_1 = np.ones((len(M_plums), len(young_ratio_plums), len(M_disks), len(young_ratio_disks), len(R_plums), len(R_disks), len(iss), len(PAs), len(aspect_ratios)))*1e100
k2s_2 = np.ones((len(M_plums), len(young_ratio_plums), len(M_disks), len(young_ratio_disks), len(R_plums), len(R_disks), len(iss), len(PAs), len(aspect_ratios)))*1e100
k2s_3 = np.ones((len(M_plums), len(young_ratio_plums), len(M_disks), len(young_ratio_disks), len(R_plums), len(R_disks), len(iss), len(PAs), len(aspect_ratios)))*1e100
k2s_4 = np.ones((len(M_plums), len(young_ratio_plums), len(M_disks), len(young_ratio_disks), len(R_plums), len(R_disks), len(iss), len(PAs), len(aspect_ratios)))*1e100

k2_1_min = 1e100
k2_2_min = 1e100
k2_3_min = 1e100
k2_4_min = 1e100
k2_min = 1e100


t0 = time.time()
ifu_cont, ifu_CO, objb = f_for_ifus(None, M_plums[0], young_ratio_plums[0], M_disks[0], young_ratio_disks[0], iss[0], PAs[0], R_plums[0], R_disks[0], aspect_ratios[0])
losvd_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocity_dispersions[1],0)
losv_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocities[1],0)
k2_1 = np.sum((ifu_cont-cube_cont)**2)
k2_2 = np.sum((ifu_CO-cube_CO)**2)
k2_3 = np.sum((losv_mod-losv)**2)
k2_4 = np.sum((losvd_mod-losvd)**2)
t1 = time.time()

eta = (t1-t0)*np.product(np.shape(k2s_1))
print("Estimated time:", eta, "("+str(np.product(np.shape(k2s_1)))+' x '+str(t1-t0)+')')
print("Shape:", np.shape(k2s_1))
#%%
for i_M_plum in tqdm(range(len(M_plums))):
    M_plum = M_plums[i_M_plum]
    for i_young_ratio_plum in range(len(young_ratio_plums)):
        young_ratio_plum = young_ratio_plums[i_young_ratio_plum]
        for i_M_disk in tqdm(range(len(M_disks))):
            M_disk = M_disks[i_M_disk]
            for i_young_ratio_disk in range(len(young_ratio_disks)):
                young_ratio_disk = young_ratio_disks[i_young_ratio_disk]
                for i_R_plum in range(len(R_plums)):
                    R_plum = R_plums[i_R_plum]
                    for i_R_disk in range(len(R_disks)):
                        R_disk = R_disks[i_R_disk]
                        for i_i in range(len(iss)):
                            i = iss[i_i]
                            for i_PA in range(len(PAs)):
                                PA = PAs[i_PA]
                                for i_aspect_ratio in range(len(aspect_ratios)):
                                    aspect_ratio = aspect_ratios[i_aspect_ratio]
                                    ifu_cont, ifu_CO, objb = f_for_ifus(None, M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio)
                                    losvd_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocity_dispersions[1],0)
                                    losv_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocities[1],0)
                                    k2_1 = np.sum((ifu_cont-cube_cont)**2)
                                    k2_2 = np.sum((ifu_CO-cube_CO_EW)**2)
                                    k2_3 = np.sum((losv_mod-losv)**2)
                                    k2_4 = np.sum((losvd_mod-losvd)**2)
                                    k2s_1[i_M_plum, i_young_ratio_plum, i_M_disk, i_young_ratio_disk, i_R_plum, i_R_disk, i_i, i_PA, i_aspect_ratio] = k2_1
                                    k2s_2[i_M_plum, i_young_ratio_plum, i_M_disk, i_young_ratio_disk, i_R_plum, i_R_disk, i_i, i_PA, i_aspect_ratio] = k2_2
                                    k2s_3[i_M_plum, i_young_ratio_plum, i_M_disk, i_young_ratio_disk, i_R_plum, i_R_disk, i_i, i_PA, i_aspect_ratio] = k2_3
                                    k2s_4[i_M_plum, i_young_ratio_plum, i_M_disk, i_young_ratio_disk, i_R_plum, i_R_disk, i_i, i_PA, i_aspect_ratio] = k2_4

#%%

mini_k1 = np.min(k2s_1)
k2s_1_normed = k2s_1/mini_k1
mini_k2 = np.min(k2s_2)
k2s_2_normed = k2s_2/mini_k2
mini_k3 = np.min(k2s_3)
k2s_3_normed = k2s_3/mini_k3
mini_k4 = np.min(k2s_4)
k2s_4_normed = k2s_4/mini_k4
k2s_normed = k2s_1_normed+k2s_2_normed+k2s_3_normed+k2s_4_normed

argmin = np.unravel_index(np.argmin(k2s_normed), np.shape(k2s_normed))

print("Shape:", np.shape(k2s_1))
print("Argmin: ", argmin)
best_M_plum = M_plums[argmin[0]]
best_young_ratio_plum = young_ratio_plums[argmin[1]]
best_M_disk = M_disks[argmin[2]]
best_young_ratio_disk = young_ratio_disks[argmin[3]]
best_R_plum = R_plums[argmin[4]]
best_R_disk = R_disks[argmin[5]]
best_i = iss[argmin[6]]
# best_i = 0.9*np.pi/2
best_PA = PAs[argmin[7]]
best_aspect_ratio = aspect_ratios[argmin[8]]

print("M_plum:", np.log10(best_M_plum))
print("M_plum_young:", np.log10(best_M_plum*young_ratio_plum), '('+str(int(best_young_ratio_plum*100))+' prct)')
print("M_disk:", np.log10(best_M_disk))
print("M_disk_young:", np.log10(best_M_disk*young_ratio_disk), '('+str(int(best_young_ratio_disk*100))+' prct)')
print("i:", 180*best_i/np.pi)
print("PA:", 180*best_PA/np.pi)
print("R_plum:", best_R_plum)
print("R_disk:", best_R_disk)
print("Aspect ratio:", best_aspect_ratio)
best_ifu_cont, best_ifu_CO, best_obj = f_for_ifus(None, best_M_plum, best_young_ratio_plum, best_M_disk, best_young_ratio_disk, best_i, best_PA, best_R_plum, best_R_disk, best_aspect_ratio)
best_losvd_mod = np.mean(best_obj.masses[1]/np.mean(best_obj.masses[1])*best_obj.x_velocity_dispersions[1],0)
best_losv_mod = np.mean(best_obj.masses[1]/np.mean(best_obj.masses[1])*best_obj.x_velocities[1],0)
im_cont = np.sum(cube_cont, 0)
# i, j = np.unravel_index(np.argmin(np.sum((best_ifu_CO-cube_CO_1)**2, 0)/abs(np.median(cube_CO_1, 0))), np.shape(im_cont))
i, j = 21, 21
spec_CO = cube_CO_EW[:, i, j]#np.mean(cube_CO_1, (1,2))
fig, axs = plt.subplots(4, 2, figsize=(5,10))

axs[0, 0].imshow(np.sum(best_ifu_cont, 0), norm=LogNorm(vmin=np.min(im_cont), vmax=np.max(im_cont)), aspect="auto")
axs[0, 1].imshow(np.sum(cube_cont, 0), norm=LogNorm(vmin=np.min(im_cont), vmax=np.max(im_cont)), aspect="auto")
axs[1, 0].plot(gaussian_filter(best_ifu_CO[:, i, j], 3))
axs[1, 1].plot(gaussian_filter(cube_CO_EW[:, i, j], 3))
axs[2, 0].imshow(best_losv_mod, aspect="auto", vmin=np.min(losv), vmax=np.max(losv))
axs[2, 1].imshow(losv, aspect="auto", vmin=np.min(losv), vmax=np.max(losv))
axs[3, 0].imshow(best_losvd_mod, aspect="auto", vmin=np.min(losvd), vmax=np.max(losvd))
axs[3, 1].imshow(losvd, aspect="auto", vmin=np.min(losvd), vmax=np.max(losvd))
plt.tight_layout()

#%%

#%%
        
def f_for_plot(_,  M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio):
    ifu_cont, ifu_CO, objb = f_for_ifus(None, M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio)
    losvd_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocity_dispersions[1],0)
    losv_mod = np.mean(objb.masses[1]/np.mean(objb.masses[1])*objb.x_velocities[1],0)   
    return ifu_cont/mini_k1**0.5, ifu_CO/mini_k2**0.5, losv_mod/mini_k3**0.5, 2*losvd_mod/mini_k4**0.5

def f_for_fit(_,  M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio):
    # print( M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio)
    c, C, v, d = f_for_plot(_, M_plum, young_ratio_plum, M_disk, young_ratio_disk, i, PA, R_plum, R_disk, aspect_ratio)
    return np.concatenate([c.flatten(), C.flatten(), v.flatten(), d.flatten()])

t0 = time.time()

obss = [cube_cont/mini_k1**0.5, cube_CO/mini_k2**0.5, losv/mini_k3**0.5, 2*losvd/mini_k4**0.5]
obs = np.concatenate([obss[0].flatten(), obss[1].flatten(), obss[2].flatten(), obss[3].flatten()])

p0 = [best_M_plum, best_young_ratio_plum, best_M_disk, best_young_ratio_disk, best_i, best_PA, best_R_plum, best_R_disk, best_aspect_ratio]
bounds = ([1e7, 0.01, 1e9, 0.01, 50*np.pi/180, 210*np.pi/180, 25, 200, 0.1], [5e9, 0.2, 1e11, 0.2, 80*np.pi/180, 260*np.pi/180, 80, 1000, 3])
p_fit, c_fit = curve_fit(f_for_fit, None, np.concatenate([obs]).flatten(), p0 = p0, bounds=bounds, maxfev=100000)

t1 = time.time()
print(t1-t0)
#%%

best_M_plum_fit, best_young_ratio_plum_fit, best_M_disk_fit, best_young_ratio_disk_fit, best_i_fit, best_PA_fit, best_R_plum_fit, best_R_disk_fit, best_aspect_ratio_fit = p_fit

print("M_plum:", np.log10(best_M_plum_fit))
print("M_plum_young:", np.log10(best_M_plum_fit*best_young_ratio_plum_fit), '('+str(int(best_young_ratio_plum_fit*100))+' prct)')
print("M_disk:", np.log10(best_M_disk_fit))
print("M_disk_young:", np.log10(best_M_disk_fit*best_young_ratio_disk_fit), '('+str(int(best_young_ratio_disk_fit*100))+' prct)')
print("i:", 180*best_i_fit/np.pi)
print("PA:", 180*best_PA_fit/np.pi)
print("R_plum:", best_R_plum_fit)
print("R_disk:", best_R_disk_fit)
print("Aspect ratio:", best_aspect_ratio_fit)

best_ifu_cont, best_ifu_CO, best_obj = f_for_ifus(None, best_M_plum_fit, best_young_ratio_plum_fit, best_M_disk_fit, best_young_ratio_disk, best_i_fit, best_PA_fit, best_R_plum_fit, best_R_disk_fit, best_aspect_ratio_fit)
best_losvd_mod = np.mean(best_obj.masses[1]/np.mean(best_obj.masses[1])*best_obj.x_velocity_dispersions[1],0)
best_losv_mod = np.mean(best_obj.masses[1]/np.mean(best_obj.masses[1])*best_obj.x_velocities[1],0)
im_cont = np.sum(cube_cont, 0)
# i, j = np.unravel_index(np.argmin(np.sum((best_ifu_CO-cube_CO_1)**2, 0)/abs(np.median(cube_CO_1, 0))), np.shape(im_cont))
i, j = 21, 21
spec_CO = np.mean(cube_CO_EW, (1,2))#np.mean(cube_CO_1, (1,2))

lenj = len(lam_ssp_rebinned_j_cont)
lenh = len(lam_ssp_rebinned_h_cont)
lenk = len(lam_ssp_rebinned_k_cont)

im_cont_j = np.mean(best_ifu_cont[:lenj], 0)
im_cont_h = np.mean(best_ifu_cont[lenj:lenj+lenh], 0)
im_cont_k = np.mean(best_ifu_cont[lenj+lenh:lenj+lenh+lenk], 0)

vmin = np.min([im_cont_j, im_cont_h, im_cont_k])
vmax = np.max([im_cont_j, im_cont_h, im_cont_k])

pix_scale = objj.pix_scale*shapej[2]/n_pix_1
ext05 = pix_scale*(n_pix_1-1)/2
extent = [ext05, -ext05, -ext05, ext05]


fig, axs = plt.subplots(3, 2, figsize=(10.5,13))

axs[0, 0].imshow(np.mean(best_ifu_cont[:lenj], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
axs[0, 1].imshow(np.mean(cube_cont[:lenj], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
axs[1, 0].imshow(np.mean(best_ifu_cont[lenj:lenj+lenh], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
axs[1, 1].imshow(np.mean(cube_cont[lenj:lenj+lenh], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
axs[2, 0].imshow(np.mean(best_ifu_cont[lenj+lenh:lenj+lenh+lenk], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
im = axs[2, 1].imshow(np.mean(cube_cont[lenj+lenh:lenj+lenh+lenk], 0), aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
axs[0, 0].set_aspect('equal')
axs[0, 0].set_xlabel('RA (")')
axs[0, 0].set_ylabel('$\delta$ (")')
axs[0, 1].set_aspect('equal')
axs[0, 1].set_xlabel('RA (")')
axs[0, 1].set_ylabel('$\delta$ (")')
axs[1, 0].set_aspect('equal')
axs[1, 0].set_xlabel('RA (")')
axs[1, 0].set_ylabel('$\delta$ (")')
axs[1, 1].set_aspect('equal')
axs[1, 1].set_xlabel('RA (")')
axs[1, 1].set_ylabel('$\delta$ (")')
axs[2, 0].set_aspect('equal')
axs[2, 0].set_xlabel('RA (")')
axs[2, 0].set_ylabel('$\delta$ (")')
axs[2, 1].set_aspect('equal')
axs[2, 1].set_xlabel('RA (")')
axs[2, 1].set_ylabel('$\delta$ (")')
# plt.tight_layout()
cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Flux ($W.m^-2.\mu m^{-1}$)', rotation=90)
plt.subplots_adjust(left=0.07, bottom=0.05, right=0.85, top=0.99, wspace=0.1, hspace=0.25)
plt.savefig('./PLOTS/fig_continuum.pdf')
plt.savefig('./PLOTS/fig_continuum.png')

fig, axs = plt.subplots(3, 2, figsize=(10.5,13))
axs[0, 0].plot(lam_ssp_rebinned_CO, gaussian_filter(np.mean(best_ifu_CO, (1,2)), 3))
axs[0, 1].plot(lam_ssp_rebinned_CO, gaussian_filter(np.mean(cube_CO_EW, (1,2)), 3))
axs[1, 0].imshow(best_losv_mod/1e3, aspect="auto", vmin=0.9*np.min(losv)/1e3, vmax=0.9*np.max(losv)/1e3, origin='lower', extent=extent)
losvim = axs[1, 1].imshow(losv/1e3, aspect="auto", vmin=0.9*np.min(losv)/1e3, vmax=0.9*np.max(losv)/1e3, origin='lower', extent=extent)
axs[2, 0].imshow(best_losvd_mod/1e3, aspect="auto", vmin=6e4/1e3, vmax=12e4/1e3, origin='lower', extent=extent)
losvdim = axs[2, 1].imshow(losvd/1e3, aspect="auto", vmin=6e4/1e3, vmax=12e4/1e3, origin='lower', extent=extent)
axs[0, 0].set_aspect(np.diff(axs[0, 0].get_xlim())/np.diff(axs[0, 0].get_ylim()))
axs[0, 0].set_xlabel('Wavelength ($\mu m$)')
axs[0, 0].set_ylabel('Relative intensity')
axs[0, 1].set_aspect(np.diff(axs[0, 1].get_xlim())/np.diff(axs[0, 1].get_ylim()))
axs[0, 1].set_xlabel('Wavelength ($\mu m$)')
axs[0, 1].set_ylabel('Relative intensity')
axs[1, 0].set_aspect('equal')
axs[1, 0].set_xlabel('RA (")')
axs[1, 0].set_ylabel('$\delta$ (")')
axs[1, 1].set_aspect('equal')
axs[1, 1].set_xlabel('RA (")')
axs[1, 1].set_ylabel('$\delta$ (")')
axs[2, 0].set_aspect('equal')
axs[2, 0].set_xlabel('RA (")')
axs[2, 0].set_ylabel('$\delta$ (")')
axs[2, 1].set_aspect('equal')
axs[2, 1].set_xlabel('RA (")')
axs[2, 1].set_ylabel('$\delta$ (")')
cbar1_ax = fig.add_axes([0.87, 0.395, 0.025, 0.25])
cbar1 = fig.colorbar(losvim, cax=cbar1_ax)
cbar1.set_label('LOSV ($km.s^{-1}$)', rotation=90)
cbar2_ax = fig.add_axes([0.88, 0.055, 0.025, 0.25])
cbar2 = fig.colorbar(losvdim, cax=cbar2_ax)
cbar2.set_label('LOSVD ($km.s^{-1}$)', rotation=90)
plt.subplots_adjust(left=0.07, bottom=0.05, right=0.85, top=0.99, wspace=0.1, hspace=0.25)
plt.savefig('./PLOTS/fig_losv.pdf')
plt.savefig('./PLOTS/fig_losv.png')
