#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Dynamic watershed main file.

In this file we implement the splitting algorithm for splitting nuclei
as described in 'Segmentation of Nuclei in Histopathology Images by deep 
regression of the distance map'. This algorithm is essentially a dynamic 
watershed. The main function implemented here is post_process.
"""

import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import dilation, erosion
from skimage.morphology import reconstruction, square, watershed
from skimage.measure import label, regionprops

def all_split_process(p_img, lamb, p_thresh=0.5, uint8=True):
    """
    Function that applies the post processing split to an
    given image.

    Args:
        p_img: input image, is usually a probability map
               or it can be a distance map. NxN matrix.
        lamb: Main parameter for the splitting process. It
              will only split two maximums if the link with 
              shortest depth bewteen them is less then lamb.
              An integer or float
        p_thresh: Threshold to cut p_img. All pixel of p_img
                  that are above p_thresh are considered positive. 
                  An integer or float.
        uint8: Whether the image is of type integer.
               True or False, a boolean. For the function to work
               properly p_img's type must be compatible with this
               parameter.

    Returns:
        A segmented map. Where each connected component is 
        assigned an integer.
    """
    b_img = (p_img > p_thresh) + 0
    probs_inv = invert_prob(p_img, uint8)
    h_recons = h_reconstruction_erosion(probs_inv, lamb, uint8)
    markers_probs_inv = find_maxima(h_recons, mask=b_img, uint8=uint8)
    markers_probs_inv = label(markers_probs_inv)
    ws_labels = watershed(h_recons, markers_probs_inv, mask=b_img)
    result = arrange_label(ws_labels)
    return result

def arrange_label(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background=background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = arrange_label(mat)
    return mat

def assign_wsl(label_res, wsl):
    """
    Assigns wsl to biggest connect component in the bbox around the wsl.
    """
    wsl_lb = label(wsl)
    objects = regionprops(wsl_lb)
    for obj in objects:
        x_b, y_b, x_t, y_t = obj.bbox
        val, counts = np.unique(label_res[x_b:x_t, y_b:y_t], return_counts=True)
        if 0 in val:
            coord = np.where(val == 0)[0]
            counts[coord] = 0
        if obj.label in val:
            coord = np.where(val == obj.label)[0]
            counts[coord] = 0
        best_lab = val[counts.argmax()]
        label_res[wsl_lb == obj.label] = best_lab
    return label_res

def find_maxima(img, uint8=True, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    recons = h_reconstruction_erosion(img, 1, uint8)
    res = recons - img
    if mask is not None:
        res[mask == 0] = 0
    return res

def generate_wsl(ws_):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se_3 = square(3)
    ero = ws_.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se_3)
    ero[ws_ == 0] = 0

    grad = dilation(ws_, se_3) - ero
    grad[ws_ == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad

def h_reconstruction_erosion(prob_img, h_value, uint8=True):
    """
    Performs a H minimma reconstruction via an erosion method.
    """
    h_img = np.zeros_like(prob_img) + h_value
    if uint8:
        h_img = np.minimum(h_img, 255 - prob_img) 
    seed = prob_img + h_img
    mask = prob_img
    recons = reconstruction(seed, mask, method='erosion')
    recons = recons.astype(prob_img.dtype)
    return recons

def invert_prob(img, uint8=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if uint8:
        img = img_as_ubyte(img)
        img = 255 - img
    else:
        img = img.max() - img
    return img

def post_process(prob_image, param=7, thresh=0.5):
    """
    Perform DynamicWatershedAlias with some default parameters.
    """
    if 'int' in prob_image.dtype:
        segmentation_mask = all_split_process(prob_image, param, thresh)
    else:        
        segmentation_mask = all_split_process(prob_image, param, thresh, uint8=False)
    return segmentation_mask
