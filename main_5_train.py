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
    np.save('preds_a5.npy', all_preds)
    np.save('labels_a5.npy', all_labels)

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


def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    # 假设你关心的站点位置如下（注意：这是你需要根据自己数据设置的！）
    stations = [(3, 4), (6, 7), (1, 10), (10, 2), (8, 8)]  # 举例子
    radius = 1  # 权重影响半径

    def create_weight_mask(shape, stations, radius=1, high=1.0, low=0.1):
        """
        创建加权mask：五个站点周边权重为 high，其他地方为 low
        shape: [1, 1, H, W]
        """
        mask = torch.ones(shape) * low
        for (i, j) in stations:
            i_start, i_end = max(0, i-radius), i+radius+1
            j_start, j_end = max(0, j-radius), j+radius+1
            mask[:, :, i_start:i_end, j_start:j_end] = high
        return mask

    def weighted_mae(pred, true, weight):
        return torch.mean(weight * torch.abs(pred - true))

    total_surf_loss = 0.0
    total_atmos_loss = 0.0

    for var in pred_batch.surf_vars:
        pred = pred_batch.surf_vars[var].float()
        true = true_batch.surf_vars[var].float()
        # pred.shape = [1, 1, H, W]
        weight = create_weight_mask(pred.shape, stations).to(pred.device)
        total_surf_loss += weighted_mae(pred, true, weight)

    for var in pred_batch.atmos_vars:
        pred = pred_batch.atmos_vars[var].float()
        true = true_batch.atmos_vars[var].float()
        # 默认不加权，或自行添加权重机制
        total_atmos_loss += torch.nn.functional.l1_loss(pred, true)

    div_free_loss = 0  # 目前不使用

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

    device = 'cuda:1'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except:
        evaluation(model, X_batch, y_batch, device)
        
