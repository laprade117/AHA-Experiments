import cv2
import glob
import numpy as np
from skimage import io

from venomai import preprocess

target_res = 5 # pixels per mm

raw_images = np.sort(glob.glob('data/raw/images/*'))
scales = np.zeros(len(raw_images))

# ImageJ measurements
scales[6] = 8.1658
scales[7] = 9.0962
scales[8] = 8.1193
scales[9] = 7.4744
scales[10] = 6.9975
scales[11] = 12.6250
scales[12] = 13.2384
scales[13] = 11.3198 
scales[14] = 13.0479
scales[15] = 13.8642
scales[17] = 13.0646
scales[18] = 11.9099
scales[19] = 10.9824
scales[20] = 4.3446
scales[21] = 10.7489
scales[22] = 10.7010
scales[23] = 10.7435
scales[24] = 3.3353
scales[25] = 5.2333
scales[26] = 3.6033
scales[27] = 3.8764
scales[28] = 3.9253

# From template matching algorithm
scales[0] = 11.4765
scales[1] = 11.4863
scales[2] = 11.8486
scales[3] = 11.4575
scales[4] = 11.3564
scales[5] = 11.4654
scales[16] = 11.2560

# # Run these lines below to confirm the template matching scale
# image_files = np.sort(glob.glob('data/raw/images/*.tif'))
# template_indices = [0,1,2,3,4,5,16]
# for i in template_indices:
    
#     # Load original image
#     image = io.imread(image_files[i])
    
#     # Convert to linear RGB color space
#     image = preprocess.srgb_to_linear(image)

#     # Apply automatic white balancing
#     image = preprocess.auto_white_balance(image)
    
#     # Compute pixel resolution
#     inner_area=10**2 # Real world area of the inside of the black squares
#     black_squares = preprocess.find_all_black_squares(image)
#     _, _, pixel_resolution = preprocess.compute_square_info(black_squares, inner_area=inner_area)
#     print(i, pixel_resolution)

image_files = np.sort(glob.glob('data/raw/images/*.tif'))
num_annotators = len(glob.glob('data/raw/masks/*'))

images = []
masks = []

for i in range(len(image_files)):
    
    print(f'{i}/{len(image_files)}')
    
    # Load original image
    image = io.imread(image_files[i])
    
    # Convert to linear RGB color space
    image = preprocess.srgb_to_linear(image)
    
    # Apply automatic white balancing
    image = preprocess.auto_white_balance(image)
    
    # Apply template white balancing
    if i in [0,1,2,3,4,5,16]:
        print('Template white balancing')
        inner_area=10**2
        black_squares = preprocess.find_all_black_squares(image)
        white_point, black_point, pixel_resolution = preprocess.compute_square_info(black_squares, inner_area=inner_area)
        image = preprocess.white_balance(image, white_point, black_point)
    
    # Convert back to gamma RGB color space
    image = preprocess.linear_to_srgb(image)
    
    # Rescale image to have a resolution of 6 pixels per mm
    image = preprocess.rescale_image(image, scales[i], target_res=target_res, interpolation=cv2.INTER_CUBIC)
    
    # Save preprocessed image and mask
    io.imsave(f'data/preprocessed/images/image_{i:03d}.tif', image)
    
    for j in range(num_annotators):
    
        mask = io.imread(f'data/raw/masks/{j}/mask_{i:04d}.tif')
        mask = preprocess.rescale_image(mask, scales[i], target_res=target_res, interpolation=cv2.INTER_CUBIC)
        mask = (np.round(mask / 255) * 255).astype('uint8')

        io.imsave(f'data/preprocessed/masks/{j}/mask_{i:03d}.tif', mask)