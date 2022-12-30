#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:27:50 2022

@author: pierre
"""
# from astropy.constants import G, pc

import numpy as np
import math
import time
import tqdm
import SSP.extract
from astropy.io import fits
import scipy.ndimage
from scipy.optimize import curve_fit

G = 6.67408e-11  # m3 kg-1 s-2
pc = 3.086e16  # m
M_sol = 1.988e30  # kg
year = 3600 * 24 * 365  # s


X_factor = 2e20  # cm−2 K−1 km−1 s

#%%
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def cart2cyl(mesh):
    r = np.sqrt(mesh[0] ** 2 + mesh[1] ** 2)
    theta = np.arctan2(mesh[1], mesh[0])
    z = mesh[2]
    mesh = np.array([r, theta, z])
    return mesh


def cyl2cart(mesh):
    x = mesh[0] * np.cos(mesh[1])
    y = mesh[0] * np.sin(mesh[1])
    mesh = np.array([x, y, mesh[2]])
    return mesh


def cart2sph(mesh):
    xy2 = mesh[0] ** 2 + mesh[1] ** 2
    r = np.sqrt(xy2 + mesh[1] ** 2)
    theta = np.arctan2(np.sqrt(xy2), mesh[2])
    phi = np.arctan2(mesh[1], mesh[0])
    mesh = np.array([r, theta, phi])
    return mesh


def sph2cart(mesh):
    x = mesh[0] * np.cos(mesh[1]) * np.sin(mesh[2])
    y = mesh[0] * np.sin(mesh[1]) * np.sin(mesh[2])
    z = mesh[0] * np.cos(mesh[1])
    mesh = np.array([x, y, z])
    return mesh


def compute_mesh(n_pix, pix_scale, i=0, PA=0, mode="cartesian"):
    xt = np.linspace(-n_pix / 2, n_pix / 2, n_pix) * pix_scale
    yt = xt.copy()
    zt = xt.copy()
    mesht = np.meshgrid(xt, yt, zt, indexing="ij")

    if i != 0:
        mat_i = rotation_matrix([0, 1, 0], i)
        mesht = np.dot(
            mat_i, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
        ).reshape(np.shape(mesht))
    if PA != 0:
        mat_PA = rotation_matrix([1, 0, 0], PA)
        mesht = np.dot(
            mat_PA, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
        ).reshape(np.shape(mesht))

    if mode == "cartesian":
        mesh = mesht
    if mode == "cylindrical":
        mesh = cart2cyl(mesht)
    if mode == "spherical":
        mesh = cart2cyl(mesht)

    return mesh


def rotate_mesh(mesh, i=0, PA=0, mode="obs_to_local"):
    mesht = mesh.copy()
    if mode == "obs_to_local":
        if PA != 0:
            mat_PA = rotation_matrix([1, 0, 0], -PA)
            mesht = np.dot(
                mat_PA, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
            ).reshape(np.shape(mesht))
        if i != 0:
            mat_i = rotation_matrix([0, 1, 0], -i)
            mesht = np.dot(
                mat_i, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
            ).reshape(np.shape(mesht))
    if mode == "local_to_obs":
        if i != 0:
            mat_i = rotation_matrix([0, 1, 0], i)
            mesht = np.dot(
                mat_i, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
            ).reshape(np.shape(mesht))
        if PA != 0:
            mat_PA = rotation_matrix([1, 0, 0], PA)
            mesht = np.dot(
                mat_PA, [mesht[0].flatten(), mesht[1].flatten(), mesht[2].flatten()]
            ).reshape(np.shape(mesht))
    return mesht


def recenter_im(im):
    center = scipy.ndimage.measurements.center_of_mass(np.ones(np.shape(im)))
    centroid = scipy.ndimage.measurements.center_of_mass(im)
    diff = np.array(np.around(np.array(centroid) - np.array(center)), dtype="int")
    return np.roll(im, -diff)


def compute_unit_vector(cartesian_mesh, i=0, PA=0, vector="cyl_theta"):
    mesht = cartesian_mesh.copy()
    mesh = rotate_mesh(mesht, i, PA, mode="obs_to_local")
    if vector in ["cart_x", "cart_x", "cart_z"]:
        x, y, z = mesh
        if vector == "cart_x":
            vector_field_x = x * 0 + 1
            vector_field_y = y * 0
            vector_field_z = z * 0
        if vector == "cart_y":
            vector_field_x = x * 0
            vector_field_y = y * 0 + 1
            vector_field_z = z * 0
        if vector == "cart_z":
            vector_field_x = x * 0
            vector_field_y = y * 0
            vector_field_z = z * 0 + 1
    if vector in ["cyl_R", "cyl_theta", "cyl_z"]:
        R, theta, z = cart2cyl(mesh)
        if vector == "cyl_R":
            vector_field_x = np.cos(theta)
            vector_field_y = np.sin(theta)
            vector_field_z = theta * 0
        if vector == "cyl_theta":
            vector_field_x = -np.sin(theta)
            vector_field_y = np.cos(theta)
            vector_field_z = theta * 0
        if vector == "cyl_z":
            vector_field_x = theta * 0
            vector_field_y = theta * 0
            vector_field_z = theta * 0 + 1
    if vector in ["sph_r", "sph_theta", "sph_z"]:
        r, theta, phi = cart2sph(mesh)
        if vector == "sph_r":
            vector_field_x = np.sin(theta) * np.cos(phi)
            vector_field_y = np.sin(theta) * np.sin(phi)
            vector_field_z = np.cos(theta)
        if vector == "sph_theta":
            vector_field_x = np.cos(theta) * np.cos(phi)
            vector_field_y = np.cos(theta) * np.sin(phi)
            vector_field_z = -np.sin(theta)
        if vector == "sph_z":
            vector_field_x = -np.sin(phi)
            vector_field_y = np.cos(phi)
            vector_field_z = theta * 0
    if vector == "Null":
        vector_field_x = cartesian_mesh[0] * 0
        vector_field_y = cartesian_mesh[0] * 0
        vector_field_z = cartesian_mesh[0] * 0
    if vector == "Uniform":
        vector_field_x = cartesian_mesh[0] * 0 + 1
        vector_field_y = cartesian_mesh[0] * 0
        vector_field_z = cartesian_mesh[0] * 0

    mesht = np.array([vector_field_x, vector_field_y, vector_field_z])
    mesh = rotate_mesh(mesht, i, PA, mode="local_to_obs")
    return mesh


#%%


def zero(mesh, *args, **kwargs):
    return mesh[0] * 0


def radius(mesh):
    array = np.array(mesh)
    return np.sum(array ** 2, 0) ** 0.5


def radius_cyl(mesh):
    array = np.array(mesh)
    return np.sum(array[:2] ** 2, 0) ** 0.5


def rad_vel_sin_los(mesh, v):
    array = np.array(mesh)
    R = radius_cyl(mesh)
    v_x = np.divide(-v * mesh[1], R, out=np.zeros_like(v), where=R != 0)
    return v_x


def point_source_cyl_vel(mesh, M, i=0, PA=0):
    r = radius(mesh)
    return (G * M / r) ** 0.5


def plummer_density(r, M0, a, i=0, PA=0):
    A = 3 * M0 / (4 * np.pi * a ** 3)
    B = (1 + r ** 2 / a ** 2) ** (-5 / 2)
    return A * B


def plummer_mesh_density(mesh, M0, a, i=0, PA=0):
    r = radius(mesh)
    A = 3 * M0 / (4 * np.pi * a ** 3)
    B = (1 + r ** 2 / a ** 2) ** (-5 / 2)
    return A * B


def plummer_acceleration(r, M0, a, i=0, PA=0):
    A = -G * M0
    B = -r
    C = (r ** 2 + a ** 2) ** 1.5
    return A * B / C


def plummer_velocity(r, M0, a, i=0, PA=0):
    return (r * plummer_acceleration(r, M0, a)) ** 0.5


def plummer_mesh_cyl_vel(mesh, M0, a, i=0, PA=0):
    r = radius(mesh)
    return (r * plummer_acceleration(r, M0, a)) ** 0.5


def plummer_velocity_dispersion(r, M0, a, i=0, PA=0):
    return (G * M0 / 6 / (r ** 2 + a ** 2) ** 0.5)**0.5


def plummer_mesh_velocity_dispersion(mesh, M0, a, i=0, PA=0):
    r = radius(mesh)
    return plummer_velocity_dispersion(r, M0, a, i=0, PA=0)


def miyamoto_nagai_mesh_density(mesh, M, a, b, i=0, PA=0):
    meshtpa = rotate_mesh(mesh, i, PA, mode="obs_to_local")
    R = radius_cyl(meshtpa)
    z = meshtpa[2]
    A = (
        M
        * b ** 2
        * (
            a * R ** 2
            + (a + 3 * (z ** 2 + b ** 2) ** 0.5) * (a + (z ** 2 + b ** 2) ** 0.5) ** 2
        )
    )
    B = (
        4
        * np.pi
        * (R ** 2 + (a + (z ** 2 + b ** 2) ** 0.5) ** 2) ** 2.5
        * (z ** 2 + b ** 2) ** 1.5
    )
    return A / B


def miyamoto_nagai_mesh_cyl_vel(mesh, M, a, b, i=0, PA=0):
    meshtpa = rotate_mesh(mesh, i, PA, mode="obs_to_local")
    R = radius_cyl(meshtpa)
    z = meshtpa[2]
    A = G * M
    B = (R ** 2 + (a + (z ** 2 + b ** 2) ** 0.5) ** 2) ** 1.5
    return R * (A / B) ** 0.5

def gauss_2d(mesh, amp, std_x, std_y):
    xx = mesh[0]
    yy = mesh[1]
    g2d = amp*np.exp(-(xx**2/2/std_x**2)-(yy**2/2/std_y**2))
    return g2d

def gauss_2d_flat(mesh, amp, std_x, std_y):
    xx = mesh[0]
    yy = mesh[1]
    g2d = amp*np.exp(-(xx**2/2/std_x**2)-(yy**2/2/std_y**2))
    return g2d.flatten()

x = np.arange(-50, 50, 2)
mesh = np.meshgrid(x, x, x)
mesh_2D = np.meshgrid(x, x)

a = 30

b_over_as = []
scale_heights = []
for b in tqdm.tqdm(np.arange(5, 500, 1)):
    mn = miyamoto_nagai_mesh_density(mesh, 1, a, b)
    im = mn[int(len(mn)/2)]
    p, c = curve_fit(gauss_2d_flat, mesh_2D, im.flatten(), p0 = [np.max(im), 30, 30])
    scale_height = p[1]/p[2]
    b_over_a = b/a
    b_over_as.append(b_over_a)
    scale_heights.append(scale_height)
    
def get_mn_ratio(b, a):
    return np.interp(b / a, b_over_as, scale_heights)
    
def miyamoto_nagai_mesh_cyl_vel_disp(mesh, M, a, b, i=0, PA=0):
    return (mesh[0] * 0 + 1) * get_mn_ratio(b, a)


#%%

epsilon = 1e-16


def gauss(mesh, x_0=0, y_0=0, std=1, e=1, phi=0, flux=1):
    im = np.zeros(np.shape(mesh[0]))

    xx, yy = mesh[0] - x_0, mesh[1] - y_0
    xxt = xx * np.cos(phi) + yy * np.sin(phi)
    yyt = -xx * np.sin(phi) + yy * np.cos(phi)
    r2 = xxt ** 2 + (yyt / max([epsilon, e])) ** 2

    im += np.exp(-r2 / std ** 2)

    im *= flux / (np.sum(im) + epsilon)
    return im


# def compute_mesh(n_pix, pix_scale, i=0, PA=0):
#     xt = np.linspace(-n_pix/2, n_pix/2, n_pix)*pix_scale
#     yt = xt.copy()
#     mesht = np.meshgrid(xt, yt,  indexing='ij')

#     return mesht


def get_alma_psf(filename, n_pix, pix_scale):
    hdu = fits.open(filename)
    header = hdu[0].header
    bmin = header["BMIN"]
    bmaj = header["BMAJ"]
    bpa = header["BPA"]
    mesh = compute_mesh(n_pix, pix_scale)
    psf = gauss(mesh, 0, 0, bmin / 2.355, bmaj / bmin, bpa * np.pi / 180, 1)
    return psf


def get_psf(n_pix, fwhm):
    mesh = compute_mesh(n_pix, 1)
    psf = gauss(mesh, 0, 0, fwhm / 2.355, 1, 0, 1)
    return psf


#%%


def get_row_compressor(old_dimension, new_dimension):
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row = 0
    which_column = 0
    while (
        which_row < (dim_compressor.shape[0]) and which_column < (dim_compressor.shape[1])
    ):
        if round(next_bin_break - which_column, 1) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:

            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size
    dim_compressor /= bin_size
    return dim_compressor


def get_column_compressor(old_dimension, new_dimension):
    return get_row_compressor(old_dimension, new_dimension).transpose()


def rebin(array, new_shape):
    # Note: new shape should be smaller in both dimensions than old shape
    return (
        np.mat(get_row_compressor(array.shape[0], new_shape[0]))
        * np.mat(array)
        * np.mat(get_column_compressor(array.shape[1], new_shape[1]))
    )

def rebin_1d(array, new_len):
    array_t = array.reshape((1, len(array)))
    array_rebinned = rebin(array_t, (1, new_len))
    return np.squeeze(np.asarray(array_rebinned))


def rebin_nd(array, new_shape):
    array_t = array.copy()
    for axis in range(len(new_shape)):
        new_shape_t = list(np.shape(array_t))
        new_shape_t[axis] = new_shape[axis]
        new_array_t = np.zeros(new_shape_t)
        new_dim = new_shape[axis]
        array_t = np.apply_along_axis(rebin_1d, axis, array_t, new_dim)
    return array_t

def get_raw_cont(ifu, n=10):
    ran = np.linspace(0, 1, len(ifu))
    shape = np.shape(ifu)
    ifu_ran = ran[:, None, None]*np.ones((len(ran), shape[1], shape[2]))
    med_0 = np.median(ifu[:n], 0)
    med_1 = np.median(ifu[-n:], 0)
    ifu_cont = ifu_ran*(med_1-med_0)+med_0
    return ifu_cont


# # abt = np.arange(len(ssps[10])).reshape((len(ssps[10]),1))
# # abt = lam.reshape((len(lam),1))
# # xt = ssps[10].reshape((len(ssps[10]),1))
# xt = np.mean(cube,(1,2)).reshape((len(lam),1))
# # x = rebin(xt, (1020, 1))
# # ab = rebin(abt, (1020, 1))
# for shape in np.arange(1,len(xt), 10):
#     try:
#         x = rebin_1d(xt, shape)
#         print(np.mean(x)-np.mean(xt))
#     except:
#         print(shape)
# # plt.plot(ab, x)
# # print(np.mean(x))

def deredshift(lam, spec, z, velocity=False):
    if velocity:
        z = z/299792458
    lam_deredshifted = lam/(1+z)
    spec_deredshifted = np.interp(lam, lam_deredshifted, spec)
    return spec_deredshifted

def redshift(lam, spec, z, velocity=False):
    if velocity:
        z = z/299792458
    lam_deredshifted = lam*(1+z)
    spec_deredshifted = np.interp(lam, lam_deredshifted, spec)
    return spec_deredshifted

# from sinfobj import extinct

# spec = rebin_1d(cube[:,29,32], len(lam_ssp_rebinned))
# v = np.arange(-1200, -800, 20)
# k2ss = []
# a = spec-scipy.ndimage.median_filter(spec, 50)
# ac = spec
# for ssp in tqdm(ssps_rebinned[:200:2]):
#     k2s = []
#     for v0 in v:
#         spect = deredshift(lam_ssp_rebinned, ssp, v0*1e3, velocity=True)
#         def extinction(_, a, AK):
#             ex = extinct(lam_ssp_rebinned, AK)
#             spec_ex = spect*ex
#             return a*spec_ex
#         p, c = scipy.optimize.curve_fit(extinction, spec, spec, p0=[np.mean(spec)/np.mean(specb), 0.1])
#         specb = extinction(spect, *p)
#         b = specb-scipy.ndimage.median_filter(specb, 50)
#         bc = specb
#         k2 = np.sum((a-b)**2+(ac-b)**2)
#         k2s.append(k2)
#     k2ss.append(k2s)
# plt.imshow(k2ss)
#%%
class stellar_bridge:
    def __init__(self, mesh_coordinates, components=[], n_except_losvd=-1):
        self.mesh = mesh_coordinates  # np.mesgrid(RA, dec, z indexing='ij')
        self.n_pix = np.shape(self.mesh)[-1]
        self.dl = abs(self.mesh[2][0][0][1]-self.mesh[2][0][0][0])
        self.dV = self.dl**3
        self.components = components  # [[Mass_function, cyl_velocity_function, velocity_function, velocity_vector, cyl_velocity_dispersion_function, velocity_dispersion_function, velocity_dispersion_vector, rotates, params, i, PA]]
        self.compute_grids()
        self.n_except_losvd = n_except_losvd

    def compute_grids(self):

        self.masses = []
        for component in self.components:
            self.masses.append(
                component[0](
                    self.mesh, *component[-3], i=component[-2], PA=component[-1]
                )*self.dV
            )

        self.cyl_vel = np.zeros(np.shape(self.mesh[0]))
        for component in self.components:
            self.cyl_vel += component[1](
                self.mesh, *component[-3], i=component[-2], PA=component[-1]
            )

        self.x_velocities = []
        for component in self.components:
            x_velo = np.zeros(np.shape(self.mesh[0]))
            if component[-4]:  # If rotates
                ux, uy, uz = compute_unit_vector(
                    self.mesh, i=component[-2], PA=component[-1], vector="cyl_theta"
                )
                x_velo += ux * self.cyl_vel
            velo = component[2](
                self.mesh, *component[-3], i=component[-2], PA=component[-1]
            )
            ux, uy, uz = compute_unit_vector(
                self.mesh, i=component[-2], PA=component[-1], vector=component[3]
            )
            x_velo += ux * velo
            self.x_velocities.append(x_velo)

        self.x_velocity_dispersions = []
        for component in self.components:
            x_velodisp = np.zeros(np.shape(self.mesh[0]))
            if component[-4]:  # If rotates
                ux, uy, uz = compute_unit_vector(
                    self.mesh, i=component[-2], PA=component[-1], vector="cyl_z"
                )
                x_velodisp += (
                    ux
                    * self.cyl_vel
                    * component[4](
                        self.mesh, *component[-3], i=component[-2], PA=component[-1]
                    )
                )
            velo = component[5](
                self.mesh, *component[-3], i=component[-2], PA=component[-1]
            )
            ux, uy, uz = compute_unit_vector(
                self.mesh, i=component[-2], PA=component[-1], vector=component[6]
            )
            x_velodisp += ux * velo
            self.x_velocity_dispersions.append(x_velodisp)

    def compute_x_velocities_digitized(self, bins):
        self.x_velocities_digitized = self.x_velocities.copy()
        for n in range(len(self.x_velocities_digitized)):
            x_velocity_digitized = self.x_velocities_digitized[n]
            if isinstance(bins, int):
                maxi = np.max(x_velocity_digitized)
                mini = np.max(x_velocity_digitized)
                bins = np.linspace(mini, maxi, bins + 1)
            x_velocity_digitized = np.digitize(x_velocity_digitized, bins[1:], right=False)
            self.x_velocities_digitized[n] = x_velocity_digitized

    def compute_x_velocity_dispersion_digitized(self, bins):
        self.x_velocity_dispersions_digitized = self.x_velocity_dispersions.copy()
        for n in range(len(self.x_velocity_dispersions_digitized)):
            x_velocity_dispersion_digitized = self.x_velocity_dispersions_digitized[n]
            if isinstance(bins, int):
                maxi = np.max(x_velocity_dispersion_digitized)
                mini = np.max(x_velocity_dispersion_digitized)
                bins = np.linspace(mini, maxi, bins + 1)
            x_velocity_dispersion_digitized = np.digitize(
                x_velocity_dispersion_digitized, bins[1:], right=False
            )
            self.x_velocity_dispersions_digitized[n] = x_velocity_dispersion_digitized
            
    def compute_ifu(self, spec_cubes, stds, vs, force_digitalization = False):
        try:
            x_velocities_digitized = self.x_velocities_digitized
        except:
            self.compute_x_velocities_digitized(vs)
            x_velocities_digitized = self.x_velocities_digitized
        try:
            x_velocity_dispersions_digitized = self.x_velocity_dispersions_digitized
        except:
            self.compute_x_velocity_dispersion_digitized(stds)
            x_velocity_dispersions_digitized = self.x_velocity_dispersions_digitized
        shape = np.shape(self.masses)
        shape_specs = np.shape(spec_cubes)
        final_ifu = np.zeros((shape_specs[-1], *shape[2:]))
        for k_compo in range(shape[0]):
            megaifu_t = np.zeros((shape_specs[-1], *shape[1:]))
            mass = self.masses[k_compo]
            x_velocity_digitized = x_velocities_digitized[k_compo]
            if k_compo != self.n_except_losvd:
                x_velocity_dispersion_digitized = x_velocity_dispersions_digitized[k_compo]
            else:
                x_velocity_dispersion_digitized = x_velocity_dispersions_digitized[k_compo]*0
            spec_cube = spec_cubes[k_compo]
            t0 = time.time()
            # spec_uniques = []
            # idx_vel_unique = np.unique(x_velocity_digitized)
            # idx_std_unique = np.unique(x_velocity_dispersion_digitized)
            # for id_v in idx_vel_unique:
            #     for id_s in idx_std_unique:
            #         mask1 = (x_velocity_digitized == id_v)
            #         mask2 = (x_velocity_dispersion_digitized == id_s)
            #         mask = mask1*mask2
            #         megaifu_t[:, mask] += spec_cube[id_s, id_v][:,None]
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        idx_std = x_velocity_dispersion_digitized[i,j,k]
                        idx_vel = x_velocity_digitized[i,j,k]
                        # print(i, j, k, idx_std, idx_vel)
                        # print(np.shape(megaifu_t), np.shape(spec_cube), np.shape(spec_cubes))
                        megaifu_t[:,i,j,k] = spec_cube[idx_std, idx_vel]
                # print(idx_std, idx_vel)
            t1 = time.time()
            final_ifu += np.sum(mass*megaifu_t,1)
        return final_ifu
    
                
    # def compute_vel_maps(self, spec_cubes, stds, vs, force_digitalization = False):
    #     try:
    #         x_velocities_digitized = self.x_velocities_digitized
    #     except:
    #         self.compute_x_velocities_digitized(vs)
    #         x_velocities_digitized = self.x_velocities_digitized
    #     try:
    #         x_velocity_dispersions_digitized = self.x_velocity_dispersions_digitized
    #     except:
    #         self.compute_x_velocity_dispersion_digitized(stds)
    #         x_velocity_dispersions_digitized = self.x_velocity_dispersions_digitized
    #     shape = np.shape(self.masses)
    #     luminosities = np.zeros(shape)
    #     vel_map = np.average(x_velocities_digitized, axis=0, weights=luminosities)
    #     for k_compo in range(shape[0]):
    #         megaifu_t = np.zeros((shape_specs[-1], *shape[1:]))
    #         mass = self.masses[k_compo]
    #         x_velocity_digitized = x_velocities_digitized[k_compo]
    #         x_velocity_dispersion_digitized = x_velocity_dispersions_digitized[k_compo]
    #         spec_cube = spec_cubes[k_compo]
    #         t0 = time.time()
    #         # spec_uniques = []
    #         # idx_vel_unique = np.unique(x_velocity_digitized)
    #         # idx_std_unique = np.unique(x_velocity_dispersion_digitized)
    #         # for id_v in idx_vel_unique:
    #         #     for id_s in idx_std_unique:
    #         #         mask1 = (x_velocity_digitized == id_v)
    #         #         mask2 = (x_velocity_dispersion_digitized == id_s)
    #         #         mask = mask1*mask2
    #         #         megaifu_t[:, mask] += spec_cube[id_s, id_v][:,None]
    #         for i in range(shape[1]):
    #             for j in range(shape[2]):
    #                 for k in range(shape[3]):
    #                     idx_std = x_velocity_dispersion_digitized[i,j,k]
    #                     idx_vel = x_velocity_digitized[i,j,k]
    #                     megaifu_t[:,i,j,k] = spec_cube[idx_std, idx_vel]
    #         t1 = time.time()
    #         print(np.sum(mass))
    #         print(np.sum(megaifu_t))
    #         final_ifu += np.sum(mass*megaifu_t,1)
    #     return final_ifu
    

    def compute_photometry_map(self, band="None"):
        if band == "None":
            map_2D = self.compute_map_mass()
        if band in self.stellar_bands:
            map_2D = self.compute_stellar_map(band)
        if band in self.molec_bands:
            map_2D = self.compute_molec_mass_map(band)
        return map_2D

    def compute_map_mass(self):
        map_2D = np.zeros((self.n_pix, self.n_pix))
        for mass in [self.NSC_mass, self.CND_mass, self.stellar_disk_mass]:
            map_2D += np.sum(mass, 0)
        return map_2D

    def compute_stellar_map(self, band):
        map_2D = np.zeros((self.n_pix, self.n_pix))
        for stellar_content in [
            [self.NSC_mass, self.age_NSC, self.Av_NSC],
            [self.stellar_disk_mass, self.age_stellar_disk, self.Av_stellar_disk],
        ]:
            mass_map_2D = np.sum(stellar_content[0], 0)
            mass_to_flux = SSP.extract.SSP_flux(
                stellar_content[2], stellar_content[1], band
            )
            map_2D += mass_map_2D * mass_to_flux
        return map_2D / 4 / np.pi / self.distance_meters ** 2

    def compute_molec_mass_map(self, band):
        map_2D = np.zeros((self.n_pix, self.n_pix))
        for molec_content in [self.CND_mass, self.stellar_disk_molecular_mass]:
            mass_map_2D = np.sum(molec_content, 0)
            mass_to_flux = 1
            map_2D += mass_map_2D * mass_to_flux
        return map_2D

    def compute_stellar_velocity_map(self, band):
        vel_map_2D = np.zeros((self.n_pix, self.n_pix))
        mass_map_2D = np.zeros((self.n_pix, self.n_pix))
        stellar_contents = [
            [self.NSC_mass, self.NSC_cyl_vel * 0, 0, 0, self.age_NSC, self.Av_NSC],
            [
                self.stellar_disk_mass,
                self.total_cyl_vel,
                self.i_stellar_disk,
                self.PA_stellar_disk,
                self.age_stellar_disk,
                self.Av_stellar_disk,
            ],
        ]
        for stellar_content in stellar_contents:
            mass_to_flux = SSP.extract.SSP_flux(
                stellar_content[5], stellar_content[4], band
            )
            mass_map_2D += np.sum(stellar_content[0], 0) * mass_to_flux
            vx, vy, vz = (
                compute_unit_vector(
                    self.mesh_0, i=stellar_content[2], PA=stellar_content[3]
                )
                * stellar_content[1]
            )
            vel_map_2D += np.sum(vx * stellar_content[0] * mass_to_flux, 0)
        vel_map_2D /= mass_map_2D
        return vel_map_2D

    def compute_molecular_velocity_map(self):
        vel_map_2D = np.zeros((self.n_pix, self.n_pix))
        mass_map_2D = np.zeros((self.n_pix, self.n_pix))
        molecular_contents = [
            [self.CND_mass, self.total_cyl_vel, self.i_CND, self.PA_CND],
            [
                self.stellar_disk_molecular_mass,
                self.total_cyl_vel,
                self.i_stellar_disk,
                self.PA_stellar_disk,
            ],
        ]
        for molecular_content in molecular_contents:
            mass_map_2D += np.sum(molecular_content[0], 0)
            vx, vy, vz = (
                compute_unit_vector(
                    self.mesh_0, i=molecular_content[2], PA=molecular_content[3]
                )
                * molecular_content[1]
            )
            vel_map_2D += np.sum(vx * molecular_content[0], 0)
        vel_map_2D /= mass_map_2D
        return vel_map_2D

    def compute_stellar_velocity_dispersion_map(self, band):
        vel_disp_map_2D = np.zeros((self.n_pix, self.n_pix))
        mass_map_2D = np.zeros((self.n_pix, self.n_pix))
        vmap = self.total_cyl_vel
        stellar_contents = [
            [
                self.NSC_mass,
                self.NSC_velocity_dispersion,
                0,
                0,
                "cart_x",
                self.age_NSC,
                self.Av_NSC,
            ],
            [
                self.stellar_disk_mass,
                self.stellar_disk_velocity_dispersion,
                self.i_stellar_disk,
                self.PA_stellar_disk,
                "cyl_z",
                self.age_stellar_disk,
                self.Av_stellar_disk,
            ],
        ]
        for stellar_content in stellar_contents:
            mass_to_flux = SSP.extract.SSP_flux(
                stellar_content[6], stellar_content[5], band
            )
            mass_map_2D += np.sum(stellar_content[0], 0) * mass_to_flux
            vx, vy, vz = (
                compute_unit_vector(
                    self.mesh_0,
                    i=stellar_content[2],
                    PA=stellar_content[3],
                    vector=stellar_content[4],
                )
                * stellar_content[1]
            )
            vel_disp_map_2D += (
                np.sum(abs(vx) * stellar_content[0] * mass_to_flux, 0) ** 2
            )
        vx, vy, vz = (
            compute_unit_vector(
                self.mesh_0,
                i=stellar_content[2],
                PA=stellar_content[3],
                vector="cyl_theta",
            )
            * vmap
        )
        vel_disp_map_2D += np.var(vx * stellar_content[0] * mass_to_flux, 0)
        vel_disp_map_2D = vel_disp_map_2D ** 0.5
        vel_disp_map_2D /= mass_map_2D
        return vel_disp_map_2D

    def compute_molecular_velocity_dispersion_map(self):
        vel_disp_map_2D = np.zeros((self.n_pix, self.n_pix))
        mass_map_2D = np.zeros((self.n_pix, self.n_pix))
        vmap = self.total_cyl_vel
        molecular_contents = [
            [
                self.CND_mass,
                self.CND_velocity_dispersion,
                self.i_CND,
                self.PA_CND,
                "cyl_z",
            ],
            [
                self.stellar_disk_molecular_mass,
                self.stellar_disk_velocity_dispersion,
                self.i_stellar_disk,
                self.PA_stellar_disk,
                "cyl_z",
            ],
        ]
        for molecular_content in molecular_contents:
            mass_map_2D += np.sum(molecular_content[0], 0)
            vx, vy, vz = (
                compute_unit_vector(
                    self.mesh_0,
                    i=molecular_content[2],
                    PA=molecular_content[3],
                    vector=molecular_content[4],
                )
                * molecular_content[1]
            )
            vel_disp_map_2D += np.sum(abs(vx) * molecular_content[0], 0) ** 2
            vx, vy, vz = (
                compute_unit_vector(
                    self.mesh_0,
                    i=molecular_content[2],
                    PA=molecular_content[3],
                    vector="cyl_theta",
                )
                * vmap
            )
            vel_disp_map_2D += np.var(vx * molecular_content[0], 0)
        vel_disp_map_2D = vel_disp_map_2D ** 0.5
        vel_disp_map_2D /= mass_map_2D
        return vel_disp_map_2D


#%%

# from scipy.optimize import curve_fit

# def gauss1d(x, a, x0, std):
#     return a*np.exp(-(x-x0)**2/(2*std**2))


# x = np.linspace(-200, 200, 201)

# mesh = np.meshgrid(x, x, x)

# widths = []
# for b in np.arange(2, 200, 1):
#     print(b)
#     width = []
#     disk = miyamoto_nagai_mesh_density(mesh, 1, 50, b)
#     im = disk[51]
#     for k in range(len(im)):
#         prof = im[k]
#         std0 = np.sum(prof)/np.max(prof)
#         p, c = curve_fit(gauss1d, x, prof, p0 = [np.max(prof), x[np.argmax(prof)], std0])
#         width.append(p[2])
#     widths.append(width)

"""
n_pix = 51
pix_scale = 1*pc
i = 0*np.pi/6
PA = 0*np.pi/3

R_NSC = 10*pc
M_NSC = 10**7.5*M_sol
age_NSC = 4e7
Av_NSC = 0
NSC_params = [R_NSC, M_NSC, age_NSC, Av_NSC]

R_CND = 5*pc
z_CND = 2*pc
M_CND = 1e3*M_sol
i_CND = 0.5*np.pi/2
PA_CND = np.pi/4
CND_params = [R_CND, z_CND, M_CND, i_CND, PA_CND]

R_stellar_disk = 50*pc
z_stellar_disk = 20*pc
M_stellar_disk = 10**9*M_sol
age_stellar_disk = 1e8
Av_stellar_disk = 0
i_stellar_disk = 0.5*np.pi/2
PA_stellar_disk = 0.4*np.pi/2
molecular_mass_stellar_disk = 10**5.5*M_sol
stellar_disk_params = [R_stellar_disk, M_stellar_disk, z_stellar_disk, age_stellar_disk, Av_stellar_disk, i_stellar_disk, PA_stellar_disk, molecular_mass_stellar_disk]

test_object = bridge()
test_object.set_distance_Mpc(12.8)
test_object.add_NSC_params(NSC_params)
test_object.add_CND_params(CND_params)
test_object.add_stellar_disk_params(stellar_disk_params)
test_object.compute_grids(n_pix, pix_scale)


im_star_J = test_object.compute_photometry_map(band='J')
im_star_H = test_object.compute_photometry_map(band='H')
im_star_K = test_object.compute_photometry_map(band='K')
im_mol_CO = test_object.compute_molec_mass_map(band='CO')
vel_map_star = test_object.compute_stellar_velocity_map(band='K')
vel_map_mol = test_object.compute_molecular_velocity_map()
vel_disp_star = test_object.compute_stellar_velocity_dispersion_map(band='K')
vel_disp_mol = test_object.compute_molecular_velocity_dispersion_map()


import matplotlib.pyplot as plt
plt.figure()
plt.imshow(im_star_J)
plt.figure()
plt.imshow(im_star_H)
plt.figure()
plt.imshow(im_star_K)
plt.figure()
plt.imshow(im_mol_CO)
plt.figure()
plt.imshow(vel_map_star)
plt.figure()
plt.imshow(vel_map_mol)
plt.figure()
plt.imshow(vel_disp_star)
plt.figure()
plt.imshow(vel_disp_mol)

"""
