"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/stable/guides.html#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations
import numpy
from skimage.io import imread
from napari.types import LayerDataTuple

def make_cell1_AP1_data() -> LayerDataTuple:
    """Generates an image"""
    url_data = r'https://github.com/zoccoler/metroid/raw/master/Data/Cell1/videos_AP/vid1.tif'
    video_AP1 = imread(url_data)

    return [(video_AP1, {"name": "cell1_video_AP1"})]

def make_cell1_EP1_data() -> LayerDataTuple:
    """Generates an image"""
    url_data = r'https://github.com/zoccoler/metroid/raw/master/Data/Cell1/video_EP/vid3.tif'
    video_EP1 = imread(url_data)

    return [(video_EP1, {"name": "cell1_video_EP1"})]
