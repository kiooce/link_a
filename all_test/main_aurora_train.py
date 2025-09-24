import os

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from utils import process_excel2, form_aurora_batch
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

def main(model, X_batch, y_batch, device):
    print("starting training...")

    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 1000

    for epoch in range(num_epochs):
        for i in range(len(X_batch)):

            X, y = X_batch[i], y_batch[i]
            pred = model.forward(X)

            for var in pred.surf_vars:
                if torch.isnan(pred.surf_vars[var]).any():
                    raise("NaN detected in pred.surf")
        
            #for var in pred.atmos_vars:
            #    if torch.isnan(pred.atmos_vars[var]).any():
            #        raise("NaN detected in pred.atmos")

            # Zero the gradients
            optimizer.zero_grad()

            lat, lon = X.metadata.lat, X.metadata.lon

            pred = pred.to(device)
            y = y.to(device)

            loss = aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to stabilize training

            optimizer.step()

        if epoch % 1 == 0:
            # Print the loss for every 10 epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    evaluation(model, X_batch, y_batch, device)


def evaluation(model, X_batch, y_batch, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for i in range(len(X_batch)):
            X, y = X_batch[i], y_batch[i]
            pred = model.forward(X)

            # 最简单的方法应该是把Batch转为tensor, 然后用mse_loss计算
            # shape => [1, 7, 2, 2].repeat(1, 1, 6, 6)
            # "2t", "pm10", "pm25", "so2", "no2", "o3", "q"
            pred_tensor = torch.cat([pred.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]), 
                                    pred.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]), 
                                    pred.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),
                                    pred.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]), 
                                    pred.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]), 
                                    pred.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),
                                    pred.atmos_vars['q'][:,:,:,:,:].reshape([1, 1, 144])],
                                    dim=1)

            y_tensor = torch.cat([y.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]), 
                                y.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]), 
                                y.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),
                                    y.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]), 
                                    y.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]), 
                                    y.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),
                                    y.atmos_vars['q'][:,:,:,:,:].reshape((1, 1, 144))],
                                    dim=1)
            
            # Append predictions and labels
            all_preds.append(pred_tensor.to(device))
            all_labels.append(y_tensor.to(device))

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    mse = F.mse_loss(all_preds, all_labels)

    all_preds = all_preds.cpu().detach().numpy()
    all_labels = all_labels.cpu().detach().numpy()
    np.save('all_preds_aurora.npy', all_preds)
    np.save('all_labels_aurora.npy', all_labels)

    print(f'Mean Squared Error: {mse.item()}')


# Loss for divergence-free constraint (as part of L_PHYS)
def divergence_free_loss(pred_u, pred_v, lat, lon):
    # Compute spatial gradients (finite difference approximation or using torch's autograd)
    dudy, dudx = torch.gradient(pred_u, lat, lon)
    dvdy, dvdx = torch.gradient(pred_v, lat, lon)
    
    # Compute divergence
    divergence = dudx + dvdy
    
    # L1 loss to penalize non-zero divergence (want a divergence-free wind field)
    div_loss = torch.mean(torch.abs(divergence))
    return div_loss


# Loss function combining MAE, divergence-free constraint, and other physical constraints
def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    # MAE loss between predicted and true variables
    mae_loss = torch.nn.L1Loss()
    total_surf_loss = 0.0
    total_atmos_loss = 0.0

    # Loop over all surface variables in the Batch object
    for var in pred_batch.surf_vars:
        total_surf_loss += mae_loss(pred_batch.surf_vars[var].float(), true_batch.surf_vars[var].float())

    # Loop over all atmospheric variables in the Batch object
    for var in pred_batch.atmos_vars:
        total_atmos_loss += mae_loss(pred_batch.atmos_vars[var].float(), true_batch.atmos_vars[var].float())

    # Physical loss: Enforce wind fields to be divergence-free
    # div_free_loss = divergence_free_loss(pred_batch.atmos_vars["u"].float(), pred_batch.atmos_vars["v"].float(), lat, lon)
    div_free_loss = 0 # we dont have u and v in atmos_vars

    # Additional physics-based loss (L_PHYS): Could include other physical constraints
    # Placeholder: Modify to add relevant physical constraints from the Aurora paper
    # phys_loss = compute_physical_loss(pred_batch)

    # Total loss: surface loss, atmospheric loss, and physical constraint loss
    total_loss = total_surf_loss + total_atmos_loss + reg_weight_div * div_free_loss
    return total_loss


if __name__ == '__main__':

    torch.set_num_threads(4) # limit number of threads used in torch

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    df = process_excel2(filename_weather, filename_airquality)

    X_batch, y_batch = form_aurora_batch(df)

    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "msl",  "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        # atmos_vars=("t", "q", "pm10", "pm25", "so2", "no2", "o3"),
        atmos_vars=("t", "q"),
        # patch_size=1, # 仅仅是为了解决坐标数量(lon)必须是patch_size倍数而暂时设置的
        # latent_levels=2, # Number of latent pressure levels >= 2
        autocast=True,  # Use AMP when re-training.
    )

    # Calculate mean for every column
    mean_values = df.select_dtypes(include=['number']).mean()
    # Calculate standard deviation for every column
    std_values = df.select_dtypes(include=['number']).std()

    # Normalisation means:
    locations["pm25"] = mean_values['pm25']
    locations["pm10"] = mean_values['pm10']
    locations["co"] = mean_values['co']
    locations["no2"] = mean_values['no2']
    locations["so2"] = mean_values['so2']
    locations["o3"] = mean_values['o3']

    # Normalisation standard deviations:
    scales["pm25"] = std_values['pm25']
    scales["pm10"] = std_values['pm10']
    scales["co"] = std_values['co']
    scales["no2"] = std_values['no2']
    scales["so2"] = std_values['so2']
    scales["o3"] = std_values['o3']

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

    device = 'cuda'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except:
        evaluation(model, X_batch, y_batch, device)
        
