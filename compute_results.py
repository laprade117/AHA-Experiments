import glob
import numpy as np
from skimage import io

import matplotlib.pyplot as plt

import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from monai.inferers import SlidingWindowInferer

from venomai import unet, loader, metrics, predictor, preprocess

# Get list of image files that have the mask
ind = np.array(list(np.arange(6))+[16])
image_files = np.sort(glob.glob('data/raw/images/*'))[ind]

doses = np.array([0, 1, 16, 2, 4, 8])
sorted_idx = np.argsort(doses)
doses = doses[sorted_idx]
image_files = image_files[sorted_idx]

results_area = np.zeros((6,6))
results_lum = np.zeros((6,6))
results_hau = np.zeros((6,6))

for i in range(6):

    # Load original image
    image = io.imread(image_files[i])
    image = preprocess.preprocess_image(image)
    
    # Segment using U-Net
    final_predictions = None
    for j in range(5):
        model = unet.UNet.load_from_checkpoint(f'models/unet_inference_{j}.ckpt')
        predictions, windows = predictor.predict_image(model, image, apply_preprocessing=False)
        if j == 0:
            final_predictions = predictions
        else:
            final_predictions += predictions
    predictions = final_predictions / 5
    
    # Compute area, luminance, HaU
    haus, real_areas, luminance_values, mean_rgb_values = predictor.compute_haemorrhagic_units(predictions, windows, return_stats=True)
    masks = np.round((predictions > 0.5)[:,:,:,None] * mean_rgb_values[:,None,None,:]).astype('uint8')

    results_area[i,:4] = real_areas
    results_area[i,4] = np.mean(real_areas)
    results_area[i,5] = np.std(real_areas)
    
    results_lum[i,:4] = luminance_values
    results_lum[i,4] = np.mean(luminance_values)
    results_lum[i,5] = np.std(luminance_values)
    
    results_hau[i,:4] = haus
    results_hau[i,4] = np.mean(haus)
    results_hau[i,5] = np.std(haus)
    
    # Create result figures
    fig, axs = plt.subplots(2, 4, figsize=(16,8))
    for j in range(4):
        axs[0,j].imshow(windows[j])
        axs[0,j].axis('off')
        axs[1,j].imshow(masks[j])
        axs[1,j].axis('off')

        axs[0,j].imshow(windows[j])
        axs[0,j].axis('off')
        axs[1,j].imshow(masks[j])
        axs[1,j].axis('off')

        axs[0,j].imshow(windows[j])
        axs[0,j].axis('off')
        axs[1,j].imshow(masks[j])
        axs[1,j].axis('off')
        
        axs[0,j].imshow(windows[j])
        axs[0,j].axis('off')
        axs[1,j].imshow(masks[j])
        axs[1,j].axis('off')
    plt.savefig(f"results/segmentations_{i}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(doses[i], np.mean(haus), haus)
    
# Save results
np.savetxt("results/results_area.csv", results_area, delimiter=",")
np.savetxt("results/results_lum.csv", results_lum, delimiter=",")
np.savetxt("results/results_hau.csv", results_hau, delimiter=",")