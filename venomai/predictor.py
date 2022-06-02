import cv2
import torch
import numpy as np
from scipy import ndimage
from venomai import preprocess

def get_n_largest_objects(binary_image, n=1, start=1):
    
    # Connected components
    num_features, labeled_array = cv2.connectedComponents(binary_image.astype('uint8'))

    # Get component sizes
    labels, counts = np.unique(labeled_array, return_counts=True)

    # Sort by component size
    sorted_count_idx = np.argsort(-counts)
    labels = labels[sorted_count_idx]
    counts = counts[sorted_count_idx]
    
    objects = []
    for i in range(start,n+start):
        if i > len(labels) - 1:
            continue    
        objects.append(labeled_array == labels[i])
        
    return np.array(objects)

def detect_windows(image, input_size=256):
    
    # Threshold image
    thresholded_image = (image[:,:,0] < (255 / 2)).astype('uint8')

    # Create window mask
    mask = get_n_largest_objects(thresholded_image, n=1)[0]
    mask_filled = ndimage.binary_fill_holes(mask).astype('uint8')
    mask = mask_filled - mask

    # Expand mask slighlty to catch any missing areas.
    struct = ndimage.generate_binary_structure(2,1)
    mask = ndimage.binary_dilation(mask, struct)

    # Find window masks
    window_masks = get_n_largest_objects(mask, n=4, start=1)

    # Find window centers
    centers = np.zeros((len(window_masks),2))
    for i in range(len(window_masks)):
        bbox = ndimage.find_objects(window_masks[i])[0]
        center_x = bbox[0].start + ((bbox[0].stop - bbox[0].start) / 2)
        center_y = bbox[1].start + ((bbox[1].stop - bbox[1].start) / 2)
        centers[i] = [center_x, center_y]
    centers = centers[np.argsort(centers[:,1])].astype(int)
    
    # Create array of windows
    windows = []
    half_input_size = int(input_size / 2)
    for i in range(4):
        window = image[centers[i,0]-half_input_size:centers[i,0]+half_input_size,
                       centers[i,1]-half_input_size:centers[i,1]+half_input_size]
        windows.append(window)
    
    windows = np.array(windows)
    return windows

def predict_windows(model, windows, augment=False):
    
    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    input_images = torch.tensor(np.moveaxis(windows / 255, -1, 1).astype('float32')).to(device)
    output_preds = model(input_images).cpu().detach().numpy()[:,0,:,:]
    
    return output_preds

def predict_image(model, image, augment=False, apply_preprocessing=True, return_windows=True, target_res=5):
    
    if apply_preprocessing:
        image = preprocess.preprocess_image(image, target_res=5)

    windows = detect_windows(image)
    predictions = predict_windows(model, windows)
    
    if return_windows:
        return predictions, windows
    else:
        return predictions
    
def compute_haemorrhagic_units(predictions, windows, target_res=5, return_stats=False):
    
    prediction_masks = predictions > 0.5
    
    # Compute area
    pixel_areas = np.sum(prediction_masks, axis=(1,2))
    real_areas = pixel_areas / (target_res**2)

    # Compute luminance
    mean_rgb_values = np.sum(prediction_masks[:,:,:,None] * windows, axis=(1,2)) / pixel_areas[:,None]
    mean_linear_rgb_values = preprocess.srgb_to_linear(mean_rgb_values) / 255
    luminance = np.dot(mean_linear_rgb_values, [0.2126, 0.7152, 0.0722])
    
    # Compute hau
    hau = real_areas / (10 * luminance)
    hau = np.nan_to_num(hau)
    
    if return_stats:
        return hau, real_areas, luminance, mean_rgb_values
    else:
        return hau