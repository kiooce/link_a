from utils import process_excel2, form_aurora_batch
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

filename_weather = "5站点气象数据20231001-1231.xlsx"
filename_airquality = "5站点202309-10月空气质量数据.xlsx"

df = process_excel2(filename_weather, filename_airquality)

X_batch, y_batch = form_aurora_batch(df)

model = Aurora(
    use_lora=False,
    surf_vars=("2t", "msl",  "pm10", "pm25", "so2", "no2", "o3"),
    static_vars=("lsm", "slt"),
    # atmos_vars=("t", "q", "pm10", "pm25", "so2", "no2", "o3"),
    atmos_vars=("t", "q"),
    # patch_size=1, # 仅仅是为了解决坐标数量(lon)必须是patch_size倍数而暂时设置的
    # latent_levels=2, # Number of latent pressure levels >= 2
    # autocast=True,  # Use AMP when re-training.
)

# Normalisation means:
locations["pm25_1000"] = 0.0
locations["pm10_1000"] = 0.0
locations["co_1000"] = 0.0
locations["no2_1000"] = 0.0
locations["so2_1000"] = 0.0
locations["o3_1000"] = 0.0

locations["pm25"] = 0.0
locations["pm10"] = 0.0
locations["co"] = 0.0
locations["no2"] = 0.0
locations["so2"] = 0.0
locations["o3"] = 0.0

# Normalisation standard deviations:
scales["pm25_1000"] = 1.0
scales["pm10_1000"] = 1.0
scales["co_1000"] = 1.0
scales["no2_1000"] = 1.0
scales["so2_1000"] = 1.0
scales["o3_1000"] = 1.0

scales["pm25"] = 1.0
scales["pm10"] = 1.0
scales["co"] = 1.0
scales["no2"] = 1.0
scales["so2"] = 1.0
scales["o3"] = 1.0

model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

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
np.save('all_preds_5.npy', all_preds)
np.save('all_labels_5.npy', all_labels)

print(f'Mean Squared Error: {mse.item()}')
