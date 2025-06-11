import os
import matplotlib.pyplot as plt
from pathlib import Path

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
    
    # Create directory for saving model weights and numpy arrays
    weights_dir = Path("epoch_weights")
    weights_dir.mkdir(exist_ok=True)
    npy_dir = Path("training_data")
    npy_dir.mkdir(exist_ok=True)

    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 1
    
    # Track losses and MSEs
    epoch_losses = []
    epoch_mses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        model.train()
        for i in range(len(X_batch)):
            X, y = X_batch[i], y_batch[i]
            pred = model.forward(X)

            for var in pred.surf_vars:
                if torch.isnan(pred.surf_vars[var]).any():
                    raise("NaN detected in pred.surf")

            # Zero the gradients
            optimizer.zero_grad()

            lat, lon = X.metadata.lat, X.metadata.lon

            pred = pred.to(device)
            y = y.to(device)

            loss = aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon)
            epoch_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Collect predictions and labels for MSE calculation
            with torch.no_grad():
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
                all_preds.append(pred_tensor)
                all_labels.append(y_tensor)

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(X_batch)
        epoch_losses.append(avg_epoch_loss)
        
        # Calculate MSE for the epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mse = F.mse_loss(all_preds, all_labels)
        epoch_mses.append(mse.item())
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, MSE: {mse.item():.4f}")
            
            # Save model weights with MSE in filename
            model_name = f"epoch_{epoch+1}_mse_{mse.item():.4f}.pth"
            torch.save(model.state_dict(), weights_dir / model_name)
            
            # Save numpy arrays for this epoch
            np.save(npy_dir / f'epoch_{epoch+1}_preds.npy', all_preds.cpu().numpy())
            np.save(npy_dir / f'epoch_{epoch+1}_labels.npy', all_labels.cpu().numpy())
            
            # Save training metrics
            np.save(npy_dir / 'training_losses.npy', np.array(epoch_losses))
            np.save(npy_dir / 'training_mses.npy', np.array(epoch_mses))

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
            if i == 0:
                pred_pm25 = pred.surf_vars['pm25'].squeeze().cpu().numpy()  # [T, H, W]
                plt.imshow(pred_pm25[-1])
                plt.title("Predicted PM2.5 - Last Timestep")
                plt.colorbar()
                plt.savefig("pre的问题.png")

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


def divergence_free_loss(pred_u, pred_v, lat, lon):
    dudy, dudx = torch.gradient(pred_u, lat, lon)
    dvdy, dvdx = torch.gradient(pred_v, lat, lon)
    divergence = dudx + dvdy
    div_loss = torch.mean(torch.abs(divergence))
    return div_loss


def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    mae_loss = torch.nn.L1Loss()
    total_surf_loss = 0.0
    total_atmos_loss = 0.0

    for var in pred_batch.surf_vars:
        total_surf_loss += mae_loss(pred_batch.surf_vars[var].float(), true_batch.surf_vars[var].float())

    for var in pred_batch.atmos_vars:
        total_atmos_loss += mae_loss(pred_batch.atmos_vars[var].float(), true_batch.atmos_vars[var].float())

    div_free_loss = 0

    total_loss = total_surf_loss + total_atmos_loss + reg_weight_div * div_free_loss
    return total_loss


if __name__ == '__main__':
    torch.set_num_threads(4)

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    df = process_excel2(filename_weather, filename_airquality)

    X_batch, y_batch = form_aurora_batch(df)

    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "msl",  "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        atmos_vars=("t", "q"),
        autocast=True,
    )

    mean_values = df.select_dtypes(include=['number']).mean()
    std_values = df.select_dtypes(include=['number']).std()

    locations["pm25"] = mean_values['pm25']
    locations["pm10"] = mean_values['pm10']
    locations["co"] = mean_values['co']
    locations["no2"] = mean_values['no2']
    locations["so2"] = mean_values['so2']
    locations["o3"] = mean_values['o3']

    scales["pm25"] = std_values['pm25']
    scales["pm10"] = std_values['pm10']
    scales["co"] = std_values['co']
    scales["no2"] = std_values['no2']
    scales["so2"] = std_values['so2']
    scales["o3"] = std_values['o3']
    from aurora.normalisation import locations, scales
    locations.clear()
    scales.clear()

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

    device = 'cuda'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except:
        evaluation(model, X_batch, y_batch, device)
