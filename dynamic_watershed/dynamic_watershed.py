#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Dynamic watershed main file.

In this file we implement the splitting algorithm for splitting nuclei
as described in 'Segmentation of Nuclei in Histopathology Images by deep 
regression of the distance map'. This algorithm is essentially a dynamic 
watershed. The main function implemented here is post_process.
"""

import numpy as np

from skimage.morphology import dilation, erosion
from skimage.morphology import reconstruction, square, watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max

from scipy.ndimage.morphology import distance_transform_edt

def all_split_process(p_img, lamb, p_thresh=0.5):
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
    Returns:
        A segmented map. Where each connected component is 
        assigned an integer.
    """
    b_img = (p_img > p_thresh) + 0
    probs_inv = invert_prob(p_img)
    h_recons = h_reconstruction_erosion(probs_inv, lamb)
    if lamb > p_thresh:
        # we must re add the vanishing object because of lamb-reconstruction
        h_recons = solve_vanishing_objects(probs_inv, h_recons, lamb, p_thresh)
    markers_probs_inv = find_maxima(h_recons, mask=b_img)
    markers_probs_inv = label(markers_probs_inv)
    ws_labels = watershed(h_recons, markers_probs_inv, mask=b_img)
    result = arrange_label(ws_labels)
    return result

def arrange_label(mat):
    """
    Given a labelled 2D matrix, returns a labelled 2D matrix
    where it tries to se the background to 0.
    Args:
        mat: 2-D labelled matrix.
    Returns:
        A labelled matrix. Where each connected component is 
        assigned an integer and the background is assigned 0.
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
    Assigns the watershed line to the biggest connect component 
    in the bounding box around the proposed watershed line.
    Args:
        label_res: 2-D labelled matrix on which we have just applied a
        watershed split.
        wsl: The watershed lines that were obtained by applying the
        watershed split
    Returns:
        A labelled integer matrix. Where each watershed line is assigned to one
        of the existing connected components. In particular to the biggest 
        connected component.
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

def find_maxima(img, mask=None):
    """
    Finds all local maxima from 2D image.
    Args:
        img: 2-D labelled matrix.
        mask: Whether or not to apply a mask
    Returns:
        Returns a 2-D matrix where local maxima have the value of 1
        and the rest are set 0.
    """
    res = peak_local_max(invert_prob(img.copy()), min_distance=4,
                         indices=False)
    res = res.astype(int)
    if mask is not None:
        res[mask == 0] = 0
    return res

def generate_wsl(labelled_mat):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    Args:
        labelled_mat: 2-D labelled matrix.
    Returns:
        a 2-D labelled matrix where each integer component
        cooresponds to a seperation between two objects. 
        0 refers to the backrgound.
    """
    se_3 = square(3)
    ero = labelled_mat.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se_3)
    ero[labelled_mat == 0] = 0

    grad = dilation(labelled_mat, se_3) - ero
    grad[labelled_mat == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad

def h_reconstruction_erosion(prob_img, h_value):
    """
    Performs a H minimma reconstruction via an erosion method.
    Args:
        prob_img: 2-D matrix on which to perform a h-reconstruction
                  with an erosion
        h_value: the parameter h for the h-reconstruction.
    Returns:
        A labelled matrix. Where each connected component is 
        assigned an integer and the background is assigned 0.
    """
    seed = prob_img.copy() + h_value
    mask = prob_img
    recons = reconstruction(seed, mask, method='erosion')
    recons = recons.astype(prob_img.dtype)
    return recons

def invert_prob(img):
    """
    Prepares the prob image for post-processing. We have to invert 
    the values in img. Minimums become and maximas and vice versa.
    It can convert from float -> to uint8 if needed.
    Args:
        img: 2-D matrix.
    Returns:
        The inversion of matrix img. 
    """
    img = img.max() - img
    return img

def post_process(prob_image, param=1, thresh=0.5):
    """
    Main function of packages. Applies the splitting algorithm 
    described in 'Segmentation of Nuclei in Histopathology Images by deep 
    regression of the distance map'. This algorithm is essentially a dynamic 
    watershed and can be applied to float/integer 2-D matrix.
    Args:
        prob_image: 2-D matrix.
        param: value to apply the h-reconstruction. This parameter
               can be seen as an error margin one wishes to impose
               on the split.
        thresh: value with respect to which we threshold prob_image
                into 2, background and forground. 
    Returns:
        A segmented map. Where each connected component is 
        assigned an integer.
    """
    if 'int' in str(prob_image.dtype):
        prob_image = prob_image.astype("int64")
        segmentation_mask = all_split_process(prob_image, param, thresh)
    else:        
        segmentation_mask = all_split_process(prob_image, param, thresh)
    return segmentation_mask

def solve_vanishing_objects(img, param_reconstruct, param, thresh):
    """
    Some connected components vanish because of the param-reconstruction even 
    tho some pixel values are above the thresh value. Indeed these connected 
    components are induced by a probability map where the regional maxima
    is between param and thresh. This function solves this vanishing problem
    by-introducing these connected components in the param_reconstruct image.
    Args:
        img: 2-D matrix.
        param_reconstruct: 2-D matrix which is the param-reconstruction of 
                           img.
        param: value to apply the h-reconstruction. This parameter
               can be seen as an error margin one wishes to impose
               on the split.
        thresh: value with respect to which we threshold prob_image
                into 2, background and forground. 
    Returns:
        A 2-D matrix which is essentially the corrected param_reconstruct
    """
    img_c = invert_prob(img.copy())
    img_c[param_reconstruct != param_reconstruct.max()] = 0
    img_c[img_c < thresh] = 0
    img_c[img_c > thresh] = 1
    img_c = distance_transform_edt(img_c).astype(int)
    param_reconstruct = invert_prob(param_reconstruct)
    param_reconstruct = param_reconstruct + img_c
    param_reconstruct = invert_prob(param_reconstruct)
    return param_reconstruct
