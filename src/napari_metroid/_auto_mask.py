# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:08:54 2021

@author: mazo260d
"""
import warnings
from napari.types import ImageData, LabelsData
# @napari_hook_implementation(specname="napari_get_reader")
def create_cell_mask(video: ImageData) -> LabelsData:
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    # from skimage.segmentation import clear_border
    import scipy.ndimage as sm
    # Checks if there is an existing image layer
    if video is None:
        return None

    #get video/image type
    ptype = str(video.dtype)
    # Checks if image is of integer type
    if ptype.startswith('uint'):
        pixel_depth = int(ptype[4:])
    else:
        warnings.warn("Image must be integer and non-binary",UserWarning)
        # print("Image must be of integer type")
        return None

    # Checks if single image or video (image stack)
    if len(video.shape)>2:
        # Sums pixel values element-wise until a saturation occurs

        #estimate number of pixel additions until saturation
        f0mean = np.mean(video[0])
        temp = (2**pixel_depth)//f0mean
        n_sum_til_saturation=temp.astype(int)

        f_sat = np.zeros_like(video[0],dtype='uint32')
        # b_sat = np.zeros_like(video[0],dtype='bool')

        #add first images pixel by pixel until some pixels saturate
        for j in range(n_sum_til_saturation-1):
            f_sat = np.add(f_sat,video[j])
        #Identify which pixels are overflown
        sat_values = f_sat>(2**pixel_depth)-1
        #Set overflown pixels to max value based on pixel depth
        f_sat[sat_values] = (2**pixel_depth)-1
        #Small blur
        f_sat = sm.gaussian_filter(f_sat,sigma=2)

        f_sat = f_sat.astype(video.dtype)
    else:
        f_sat = video

    thresh = threshold_otsu(f_sat)
    mask = f_sat > thresh
    #Get image dimensions
    min_dim = np.amin(mask.shape)
    max_dim = np.amax(mask.shape)
    mask = remove_small_objects(mask,(max_dim*min_dim)//10)
    # # Remove artifacts connected to image border
    # mask = clear_border(mask)
    return(mask)


    # if len(video.shape)>2: #Get only videos, single 8/16-bit images are not included
    #     if video.shape[-1]>3: # rgb images are not included (as a side-effect videos of up to 3 frames are not included)
    #         print(video.shape)
    #         mask = np.amax(video)
    #         return(mask)
    #     else:
    #         print("Erro2")
    # else:
    #     print("Error1")
