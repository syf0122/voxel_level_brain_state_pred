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
from loss import *

save_dir = 'Use your save directory here' # path to save the test results
# make save dir if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

window_size = 32 # number of frames used as input
key = 'unrel'
if key == 'test':
    test_sub = np.load(f'Use your test subjects file here') # original test subjects
    data_dir = 'Use your test data directory here'
elif key == 'train':
    test_sub = np.load(f'Use your training subjects file here') # original training subjects
    data_dir = 'Use your training data directory here'
elif key =='unrel':
    with open('Use your unrelated subjects file here') as f:
        subjects = f.readlines()
    test_sub = [str(x.strip()) for x in subjects]
    data_dir = 'Use your data directory here' # path to the preprocessed data
######################### DATA LOADER PARAMETERS #############################
time_size = 1200 # number of frames in each training example
max_size = 50 # maximum number of frames to use for prediction - 550 timesteps in each file, 500 usable instances
out_size = 10 # number of future volumes to use as target  
session_type = "REST1" # suffix in the filename for the actual data,  
downsampled = True
num_workers = 2 # changed num_workers to 0 for local, was 4 for hpc
target_shape = (48, 48, 48) # desired output shape for resizing 
transform = Resize(target_shape) # further preprocesses the data, mainly reshaping 

print(f'Use {len(test_sub)} subjects for testing.')
######################### INITIALISE DATASET ################################
test_data = fMRIDataset(data_dir,
                        data_dir, 
                        test_sub, 
                        sample_size=window_size, 
                        max_window_size=max_size, 
                        output_size=out_size, 
                        transform=transform, 
                        session_type=session_type, 
                        spatial_downsampled=downsampled,
                        time_size=time_size
                        ) 

print(f"\nDataset initialized")
print(f"Testing Dataset length: {len(test_data)} examples")
test_dataloader = DataLoader(test_data,
                             batch_size=1, 
                             shuffle=False,
                             pin_memory=False,
                             num_workers=num_workers) 

print(f"Dataloader initialized")
print(f"Testing Dataloader length: {len(test_dataloader)} batches")
print(f"Up to {len(test_data)} recordings, {len(test_dataloader)} batches to test on. That is {len(test_data)/len(test_sub)} examples per subject")
# quit()
###################### LOAD MODEL #################################
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device name: {torch.cuda.get_device_name(device)}")
loaded_model = torch.load('Use your model path here', map_location=device, weights_only=False)
if isinstance(loaded_model, nn.DataParallel):
    model = loaded_model.module
else:
    model = loaded_model
model_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {model_params:,}")
######################### Evaluation ################################
masked_mse = MaskedMSELoss().to(device)
optimised_ssim_loss = OptimizedSSIM3DLoss()
combined_loss = CombinedLoss(ssim_weight=2).to(device)
model.eval()
test_loss = []
test_mse = []
test_ssim = []
with torch.no_grad():
    progress_bar = tqdm(range(len(test_dataloader)))
    for batch_idx, (data, target, target_mask) in enumerate(test_dataloader):
        input_data = data.to(device).float()
        trg = target.to(device).float()
        trg_mask = target_mask.to(device).float()
        # Forward pass
        pred = model.forward(input_data, trg_mask)
        # Calculate losses
        batch_loss, batch_mse, batch_ssim = combined_loss(pred, trg, trg_mask, individual_losses=True)
        test_loss.append(batch_loss.item())
        test_mse.append(batch_mse.item())
        test_ssim.append(batch_ssim.item())
        progress_bar.update(1)
progress_bar.close()

# Add averaged batch loss to history
test_loss = np.array(test_loss)
test_mse = np.array(test_mse)
test_ssim = np.array(test_ssim)

print(f"Test Loss: {test_loss.mean():.6f} | MSE: {test_mse.mean():.6f} | SSIM: {test_ssim.mean():.6f}")
np.save(f'{save_dir}swin_new_downsampled_{key}_loss.npy', np.array(test_loss))
np.save(f'{save_dir}swin_new_downsampled_{key}_mse_loss.npy', np.array(test_mse))
np.save(f'{save_dir}swin_new_downsampled_{key}_ssim_loss.npy', np.array(test_ssim))