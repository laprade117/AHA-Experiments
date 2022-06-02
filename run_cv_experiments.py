import glob
import numpy as np

import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from venomai import unet, loader, metrics

def train_model(i, j, num_annotators,
                project_name, model_path, model_name, 
                full_loader, train_loader, val_loader, test_loader, 
                epochs, batch_size, lr):
    '''
    Trains a UNet model with the given parameters.
    '''

    # Setup wandb logger
    wandb_logger = WandbLogger(name=f'Run_{i}_{j}', project=project_name, log_model=False)

    # Save best model callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=model_path,
                                                       filename=f'{model_name}_{i}_{j}',
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
    del model

    model = unet.UNet.load_from_checkpoint(f'{model_path}{model_name}_{i}_{j}.ckpt')
    
    # Compute test set results per annotator
    model.set_test_metric_label('test/annotator_0')
    test_loader.dataset.evaluate(0)
    results = trainer.test(model, test_loader)
    for k in range(1, num_annotators):
        model.set_test_metric_label(f'test/annotator_{k}')
        test_loader.dataset.evaluate(k)
        results += trainer.test(model, test_loader)
    
    # Compute best loss across entire dataset
    model.set_test_metric_label('dataset')
    dataset_results = trainer.test(model, full_loader)
    
    wandb.finish()
                           
    del trainer, model
    return results, dataset_results

project_name = 'VenomAI-Haemorrhage-UNet'
model_path = 'models/'
model_name = 'unet'

epochs = 100
batch_size = 32
lr = 0.0001

num_annotators = 2
num_runs = 5
num_folds = 5

# Array containing all final results
results = np.zeros((num_runs, num_folds, num_annotators, 5))
dataset_results = np.zeros((num_runs, num_folds, 5))

for i in range(num_runs):
    
    outer_seed = np.random.randint(1000000)
    inner_seed = np.random.randint(1000000)
    
    for j in range(num_folds):
        
        # Get data loaders
        full_loader, train_loader, val_loader, test_loader = loader.create_data_loaders(i,
                                                                                        batch_size=batch_size,
                                                                                        input_size=256,
                                                                                        outer_seed=outer_seed,
                                                                                        inner_seed=inner_seed)

        run_results, dataset_run_results = train_model(i, j, num_annotators,
                                                       project_name, model_path, model_name,
                                                       full_loader, train_loader, val_loader, test_loader,
                                                       epochs, batch_size, lr)
        print(dataset_run_results)
        
        for k in range(num_annotators):
            results[i,j,k,:] = np.array(list(run_results[k].values()))
            
        dataset_results[i,j,:] = np.array(list(dataset_run_results[0].values()))
        
best_loss = total_best_loss / (num_runs * num_folds)
np.save('unet_results.npy', results)
np.save('unet_best_loss.npy', best_loss)