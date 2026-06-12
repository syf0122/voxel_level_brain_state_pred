import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset_withmask import *
from swin_trans_model import *
from loss import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device name: {torch.cuda.get_device_name(device)}")

data_dir = 'Use your data directory here' # path to the preprocessed data
save_dir = 'Use your save directory here' # path to save the model
# make save dir if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
num_sub = 100 # specifies the file containing subjects ID's
window_size = 32 # number of frames used as input
train_sub = np.load('Use your training subjects file here') # original training subjects
######################### DATA LOADER PARAMETERS #############################
time_size = 1200 # number of frames in each training example
max_size = 50 # maximum number of frames to use for prediction - 600 timesteps in each file, 550 usable instances
out_size = 10 # number of future volumes to use as target
session_type = "REST1" # suffix in the filename for the actual data
downsampled = False # whether to use the spatially downsampled data (48x48x48) or the original (96x96x96)
pin_memory = False
batch_size = 4
num_workers = 4 
spatial_dim = 96
target_shape = (spatial_dim, spatial_dim, spatial_dim) # desired output shape for resizing 
transform = Resize(target_shape) # further preprocesses the data, mainly reshaping 
# transform = None

print(f'Use {len(train_sub)} subjects for training.')
######################### INITIALISE DATASET ################################
train_data = fMRIDataset(data_dir, 
                         data_dir,
                         train_sub, 
                         sample_size=window_size, 
                         max_window_size=max_size, 
                         output_size=out_size, 
                         transform=transform, 
                         session_type=session_type, 
                         spatial_downsampled=downsampled,
                         time_size=time_size
                        ) 

print(f"\nDataset initialized")
print(f"Training Dataset length: {len(train_data)} examples")
train_dataloader = DataLoader(train_data,
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=pin_memory,
                                num_workers=num_workers) 

print(f"Dataloader initialized")
print(f"Training Dataloader length: {len(train_dataloader)} batches")
print(f"Up to {len(train_data)} examples, {len(train_dataloader)} batches to train on. That is {len(train_data)/len(train_sub)} examples per subject")
# quit()
#################### LEARNING RATE ############################
lr = 1e-4 
print(f"Learning Rate is {lr}")
###################### MASKED MODELS #################################
model_name = 'full_swin_new100' 
print(f'Training model: {model_name}')
if model_name == 'half_swin_new100':
    model = Masked_FinerUNetSwinTransformer4tspred(
        img_size=(spatial_dim, spatial_dim, spatial_dim, 32),  # (W, D, H, T)
        output_size=10,
        in_chans=1,
        embed_dim=36,
        window_size=(4, 4, 4, 4),
        first_window_size=(2, 2, 2, 4),
        patch_size=(3, 3, 3, 4),
        depths=(2, 6, 2),
        num_heads=(3, 6, 12),
        merge_divide=(2, 2, 2, 1),
        downsample="mergingv2",
    )
elif model_name == 'full_swin_new100':
    model = Masked_FinerUNetSwinTransformer4tspred(
        img_size=(spatial_dim, spatial_dim, spatial_dim, 32),  # (W, D, H, T)
        output_size=10,
        in_chans=1,
        embed_dim=36,
        window_size=(4, 4, 4, 4),
        first_window_size=(2, 2, 2, 4),
        patch_size=(6, 6, 6, 4),
        depths=(2, 6, 2),
        num_heads=(3, 6, 12),
        merge_divide=(2, 2, 2, 1),
        downsample="mergingv2",
    )
model_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {model_params:,}")
# quit()

if torch.cuda.device_count() > 1:
    if batch_size % 4 == 0:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs in parallel")
model.to(device)

masked_mse = MaskedMSELoss().to(device)
optimised_ssim_loss = OptimizedSSIM3DLoss()
combined_loss = CombinedLoss(ssim_weight=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 24
loss_hist = []
mse_loss_hist = []
ssim_loss_hist = []

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}")
    model.train()
    progress_bar = tqdm(range(len(train_dataloader)))
    epoch_loss = []
    epoch_mse = []
    epoch_ssim = []
    
    for batch_idx, (data, target, target_mask) in enumerate(train_dataloader):
        input_data = data.to(device).float()
        trg = target.to(device).float()
        trg_mask = target_mask.to(device).float()
        # Forward pass
        pred = model.forward(input_data, trg_mask)
        # Calculate losses
        batch_loss, batch_mse, batch_ssim = combined_loss(pred, trg, trg_mask, individual_losses=True)
        
        optimizer.zero_grad()  
        batch_loss.backward() 
        optimizer.step()

        # Accumulate batch loss for epoch average
        epoch_loss.append(batch_loss.item())
        epoch_mse.append(batch_mse.item())
        epoch_ssim.append(batch_ssim.item())
        progress_bar.update(1)
        
    # Add averaged batch loss to history
    avg_epoch_loss = np.mean(np.array(epoch_loss))
    avg_epoch_mse = np.mean(np.array(epoch_mse))
    avg_epoch_ssim = np.mean(np.array(epoch_ssim)) 
    loss_hist.append(epoch_loss)
    mse_loss_hist.append(epoch_mse)
    ssim_loss_hist.append(epoch_ssim)
    
    progress_bar.close()
    print(f"Training Loss: {avg_epoch_loss:.6f} | MSE: {avg_epoch_mse:.6f} | SSIM: {avg_epoch_ssim:.6f}")
    # # Save checkpoint at the end of each epoch
    # torch.save(model, f'{save_dir}{model_name}_HALF_fold-{fold+1}.pth')
    # # save the loss change history
    # np.save(f'{save_dir}{model_name}_HALF_fold-{fold+1}_loss_hist.npy', np.array(loss_hist))
    # np.save(f'{save_dir}{model_name}_HALF_fold-{fold+1}_mse_loss_hist.npy', np.array(mse_loss_hist))
    # np.save(f'{save_dir}{model_name}_HALF_fold-{fold+1}_ssim_loss_hist.npy', np.array(ssim_loss_hist))
    # print(f"Checkpoint saved for epoch {epoch+1}") 

    # Save checkpoint at the end of each epoch
    torch.save(model, f'{save_dir}{model_name}_epoch{epoch+1}.pth')
    # save the loss change history
    np.save(f'{save_dir}{model_name}_loss_hist.npy', np.array(loss_hist))
    np.save(f'{save_dir}{model_name}_mse_loss_hist.npy', np.array(mse_loss_hist))
    np.save(f'{save_dir}{model_name}_ssim_loss_hist.npy', np.array(ssim_loss_hist))
    print(f"Checkpoint saved for epoch {epoch+1}") 




