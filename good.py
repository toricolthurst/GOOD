'''
@author: Tori Colthurst

Implementation in Python of the paper GOOD:
Global Orthographic object descriptor for 3D object recognition and manipulation.
Paper Authors: S. Hamidreza Kasaei, Ana Maria Tomé, Luís Seabra Lopes, Miguel Oliveira
'''

import numpy as np
import open3d as o3d
import random
import copy
from scan_objects.helper_functions import *
import matplotlib.pyplot as plt
from scan_objects.pc_registration import execute_global_registration, execute_fast_global_registration, preprocess_point_cloud

def pca(cloud_array):

    ## principle component analysis

    m = cloud_array.shape[0]
    p_ = np.mean(cloud_array, axis=0)

    # calculate covariance matrix
    temp = cloud_array - p_
    temp = np.expand_dims(temp, axis=-1)
    covariance = np.matmul(temp, np.transpose(temp, (0, 2, 1)))
    covariance = np.sum(covariance, axis=0) / m

    # calculate PCA
    w, v = np.linalg.eig(covariance)
    idx = np.argsort(w)
    w = w[idx]
    v = v[idx]
    x, y = v[2], v[1]
    x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)

    z = np.cross(x,y)
    basis = np.transpose(np.concatenate((x,y,z), axis=0))

    return basis, p_

def calculate_s(cloud_array, t):

    ## determine sign of axes s = sx * sy

    x = cloud_array[:, 0]
    sx_pos, sx_neg = np.count_nonzero(x > t), np.count_nonzero(x < -t)
    sx = 1 if (sx_pos-sx_neg) >= 5 else -1

    y = cloud_array[:, 1]
    sy_pos, sy_neg = np.count_nonzero(y > t), np.count_nonzero(y < -t)
    sy = 1 if (sy_pos-sy_neg) >= 5 else -1

    return sx*sy

def count_and_flip(cloud, b, t):

    ## count number of points in +/- x and y axes and flip axes accordingly

    cloud_array = np.asarray(cloud.points)

    s = calculate_s(cloud_array, t)
    if s < 0:
        x, y = np.asarray([[s*int(b[0][0]),0,0]]), np.asarray([[0,s*int(b[1][1]),0]])
        z = np.cross(x,y)
        basis = np.transpose(np.concatenate((x,y,z), axis=0))
        cloud.rotate(basis)

    return cloud

def lrf(cloud, t):

    ## local reference frame

    cloud_new = o3d.geometry.PointCloud(cloud)
    cloud_array = np.asarray(cloud.points)
    basis, center = pca(cloud_array)

    cloud_new = array_to_cloud_color(cloud_array - center, cloud_new.colors)

    bbox = cloud_new.get_oriented_bounding_box()
    R = np.linalg.inv(bbox.R)
    cloud_new.rotate(R)
    bbox = cloud_new.get_oriented_bounding_box()
    cloud_new = count_and_flip(cloud_new, bbox.R, t)

    return cloud_new, center

def make_descriptor(cloud, n, scanner):

    cloud_array = np.asarray(cloud.points)
    bbox, dims = get_bbox(cloud_array)
    l = max([abs(d) for d in dims])*2
    e = 0.0001 #l/1000.0

    # callculate orthographic projections onto xy, yz, and xz planes
    M_xy, M_yz, M_xz = np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
    idx_x = (n*(cloud_array[:, 0] + l/2.0)/(l+e)).astype(int)
    idx_y = (n*(cloud_array[:, 1] + l/2.0)/(l+e)).astype(int)
    idx_z = (n*(cloud_array[:, 2] + l/2.0)/(l+e)).astype(int)

    for x,y,z in zip(idx_x,idx_y,idx_z):
        M_xy[x][y] += 1
        M_yz[y][z] += 1
        M_xz[x][z] += 1

    # normalize and flatten results into 1D arrays
    M_xy = (M_xy / np.sum(M_xy)).flatten()
    M_yz = (M_yz / np.sum(M_yz)).flatten()
    M_xz = (M_xz / np.sum(M_xz)).flatten()
    M_array = [np.expand_dims(M_xy, axis=0), np.expand_dims(M_yz, axis=0), np.expand_dims(M_xz, axis=0)]

    ## concatenate descriptors in order of entropy, then variance

    # calculate entropy
    H_xy = -np.sum(M_xy*np.where(M_xy > 0, np.log2(M_xy), 0))
    H_yz = -np.sum(M_yz*np.where(M_yz > 0, np.log2(M_yz), 0))
    H_xz = -np.sum(M_xz*np.where(M_xz > 0, np.log2(M_xz), 0))

    first = np.argmax([H_xy, H_yz, H_xz])

    # calculate variance
    i = np.asarray(range(n**2))
    mu_xy = np.sum(i*M_xy)
    var_xy = np.sum((i-mu_xy)**2*M_xy)
    mu_yz = np.sum(i*M_yz)
    var_yz = np.sum((i-mu_yz)**2*M_yz)
    mu_xz = np.sum(i*M_xz)
    var_xz = np.sum((i-mu_xz)**2*M_xz)

    if first == 0:
        remaining = [min(var_yz, var_xz)-1, var_yz, var_xz]
    elif first == 1:
        remaining = [var_xy, min(var_xy, var_xz)-1, var_xz]
    else:
        remaining = [var_xy, var_yz, min(var_xy, var_yz)-1]

    second = np.argmax(remaining)

    if first + second == 1:
        last = 2
    elif first + second == 2:
        last = 1
    else:
        last = 0

    descriptor = np.concatenate((M_array[first],M_array[second],M_array[last]), axis=0)

    return descriptor

def GOOD(cloud, n, t, scanner):

    cloud_array = np.asarray(cloud.points)
    new_cloud, center = lrf(cloud, t)

    descriptor = make_descriptor(new_cloud, n, scanner)

    return descriptor
