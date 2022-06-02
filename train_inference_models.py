import glob
import numpy as np
from skimage import io
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import wandb

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from venomai import unet, loader, metrics, predictor, preprocess

project_name = 'VenomAI-Haemorrhage-UNet-Inference'
model_path = 'models/'
model_name = 'unet_inference'

epochs = 100
batch_size = 32
lr = 0.0001

def setup_loaders(split_index, random_seed=15496):
    images, masks = loader.load_preprocessed_data()

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    i = 0
    for train_idx, val_idx in kf.split(images):
        if split_index == i:
            break
        i += 1

    val_images = [images[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]

    train_images = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]

    train_dataset = loader.UNetLoader(train_images,
                               train_masks,            
                               input_size=256,
                               training=True,
                               augment=True)
    val_dataset = loader.UNetLoader(val_images,
                             val_masks,
                             input_size=256,
                             training=False,
                             augment=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    return train_loader, val_loader

for i in range(5):

    train_loader, val_loader = setup_loaders(i)
    
    # Setup wandb logger
    wandb_logger = WandbLogger(name=f'Inference_{i}', project=project_name, log_model=False)

    # Save best model callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=model_path,
                                                       filename=f'{model_name}_{i}',
                                                       monitor="val/mcc_ce_loss",
                                                       mode="min")

    # Stochastic weight averaging callback
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.75,
                                                          swa_lrs=0.05,
                                                          annealing_epochs=10,
                                                          annealing_strategy='cos')

    # Load model
    model = unet.UNet(lr=lr, num_channels=3, num_classes=2)

    # Train model
    model.train()
    trainer = pl.Trainer(max_epochs=epochs,
                         log_every_n_steps=1,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, swa_callback],
                         gpus=1,
                         accelerator="gpu")
    trainer.fit(model, train_loader, val_loader)
    
    wandb.finish()