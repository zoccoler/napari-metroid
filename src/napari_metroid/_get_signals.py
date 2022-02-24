# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:02:34 2021

@author: mazo260d
"""
from napari.types import ImageData, LabelsData, LayerDataTuple
def build_time_vector(fr,video_shape):
    import numpy as np
    time = np.cumsum(np.ones(video_shape[0]))
    time = (time-1)/fr  
    return(time)

def get_ROIs_average_over_time(video: ImageData, label_image: LabelsData, frame_rate_info: bool=True, frame_rate: int=30):
    # import matplotlib.pyplot as plt
    import numpy as np
    label_image = label_image.data
    # from matplotlib.backends.backend_qt5agg import FigureCanvas
    if video is not None:
        if len(video.shape)>2:
            n_ROIs =  np.amax(label_image)
            ROIs_avgs = np.empty((video.shape[0],n_ROIs),'float64') #vector with size = number of regions
            for k in range(video.shape[0]): #iteration over each frame
                for j in range(n_ROIs): #iteration over each region
                    ROIs_avgs[k,j] = np.mean(video[k][label_image==j]) #video frame k, indented by ROI mask j
                    #rows are frame numbers, columns are ROI numbers
            if ((frame_rate_info==True) & (frame_rate>0)):
                time = build_time_vector(frame_rate, video.shape)
            else:
                time = np.cumsum(np.ones(video.shape[0]))
            
            return(ROIs_avgs,time)