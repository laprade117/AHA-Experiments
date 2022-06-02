import cv2
import shutil
import numpy as np
from pathlib import Path

def pad_to_input_shape(ndarray, input_shape=512, channel_axis=None, mode='constant'):
    """
    Pads input ndarray so that each dimension has a minimum length of input_size.
    """
    
    ndarray_shape = np.array(ndarray.shape)

    lower_pad = np.floor((input_shape - ndarray_shape) / 2).clip(min=0).astype(int)
    upper_pad = np.ceil((input_shape - ndarray_shape) / 2).clip(min=0).astype(int)
    padding = np.stack([lower_pad, upper_pad]).T
    
    if channel_axis is not None:
        padding[channel_axis] *= 0

    return np.pad(ndarray, padding, mode=mode)

def crop_to_input_shape(ndarray, input_shape=512):
    """
    Crops input ndarray so that each dimension has a maximum length of input_size.
    """

    ndarray_shape = np.array(ndarray.shape)

    start = np.floor((ndarray_shape - input_shape) / 2).clip(min=0).astype(int)
    end = start + input_shape
    slices = tuple([slice(start[i], end[i]) for i in range(len(ndarray_shape))])

    return ndarray[slices]

def make_categorical(data, class_values=[0,1]):
    # class_values = np.unique(data)
    data = np.moveaxis(np.array([data == v for v in class_values]), 0, -1).astype('uint8')
    return data

def extract_input_images(image, input_size=256):
    
    input_images = np.zeros((4, input_size, input_size, 3), dtype='uint8')
    
    # Blur and threshold image
    blurred_image = cv2.GaussianBlur(image[:,:,0], (9,9), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Connected components on thresholded image
    num_features, labeled_array = cv2.connectedComponents(thresholded_image)

    # Find and mask largest objects
    areas = np.array([np.sum(labeled_array == i) for i in range(num_features)])
    features = np.arange(num_features)
       
    for i in range(2,6):
        # Mask current region
        region = labeled_array == np.argsort(-areas)[i]
        region_coords = np.argwhere(region > 0)
        
        # Get bounding box
        start = np.array([np.min(region_coords[:,0]), np.min(region_coords[:,1])])
        stop = np.array([np.max(region_coords[:,0]), np.max(region_coords[:,1])])
        
        # Extract bounding box from image
        image_region = image[start[0]:stop[0], start[1]:stop[1]]
        
        # Pad input_image to input_size and at to list
        input_images[i-2] = pad_to_input_shape(image_region, (input_size,input_size,3)).astype('uint8')
        
    return input_images