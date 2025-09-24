import os

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from utils import (
    process_excel2, form_aurora_batch, plot_predictions_vs_labels_enhanced,
    specific_humidity_to_rh, uv_to_wind_speed_dir
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

def create_weather_visualization(all_preds, all_labels, converted_preds, converted_labels):
    """
    Create separate visualization for converted meteorological variables
    """
    weather_vars = {
        'Temperature (°C)': 0,
        'Humidity (%)': 1,
        'Wind Speed (m/s)': 2, 
        'Wind Direction (°)': 3,
        'Pressure (hPa)': 4,
        'Rainfall (mm)': 5
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Meteorological Variables: Predictions vs True Values', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, (var_name, var_idx) in enumerate(weather_vars.items()):
        ax = axes_flat[idx]
        
        pred_values = []
        true_values = []
        
        for i in range(len(converted_preds)):
            pred_vals = converted_preds[i][:, var_idx].flatten()
            true_vals = converted_labels[i][:, var_idx].flatten()
            
            pred_values.extend(pred_vals)
            true_values.extend(true_vals)
        
        point_indices = range(len(pred_values))
        ax.plot(point_indices, pred_values, color='#FF6B6B', linewidth=1, 
               alpha=0.7, label='Predicted')
        ax.plot(point_indices, true_values, color='#4ECDC4', linewidth=1, 
               alpha=0.7, label='True')
        
        ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Point Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        correlation = np.corrcoef(pred_values, true_values)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('meteorological_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Meteorological visualization saved: meteorological_predictions_vs_true.png")

def convert_predictions_to_physical_units(all_preds, all_labels):
    """
    Convert Aurora predictions back to physical units - FIXED VERSION
    """
    converted_preds = []
    converted_labels = []
    
    for pred_batch, label_batch in zip(all_preds, all_labels):
        pred_np = pred_batch.cpu().numpy()  # Shape: (11, 144)
        label_np = label_batch.cpu().numpy()
        
        print(f"Debug: pred_np shape = {pred_np.shape}")
        print(f"Debug: label_np shape = {label_np.shape}")
        
        n_vars, n_points = pred_np.shape  # 11 variables, 144 points
        
        # 5 meteorological variables (no rainfall)
        pred_converted = np.zeros((n_points, 5))
        label_converted = np.zeros((n_points, 5))
        
        for point_idx in range(n_points):
            # 2维数组索引: pred_np[variable_idx, point_idx]
            # Aurora输出顺序：[temp, pm10, pm25, so2, no2, o3, q, u, v, msl, co]
            # 对应索引：     [0,    1,    2,     3,   4,   5,  6, 7, 8, 9,   10]
            
            temp_pred = pred_np[0, point_idx]      # 温度
            temp_label = label_np[0, point_idx]
            q_pred = pred_np[6, point_idx]         # 比湿度
            q_label = label_np[6, point_idx]
            u_pred, v_pred = pred_np[7, point_idx], pred_np[8, point_idx]  # u,v风
            u_label, v_label = label_np[7, point_idx], label_np[8, point_idx]
            pressure_pred = pred_np[9, point_idx]  # 压力
            pressure_label = label_np[9, point_idx]
            
            # Temperature: K to °C
            pred_converted[point_idx, 0] = temp_pred - 273.15
            label_converted[point_idx, 0] = temp_label - 273.15
            
            # Specific humidity to relative humidity %
            pressure = 100000  # assume 1000 hPa = 100000 Pa
            pred_converted[point_idx, 1] = specific_humidity_to_rh(q_pred, temp_pred, pressure)
            label_converted[point_idx, 1] = specific_humidity_to_rh(q_label, temp_label, pressure)
            
            # U/V wind to speed/direction
            speed_pred, dir_pred = uv_to_wind_speed_dir(u_pred, v_pred)
            speed_label, dir_label = uv_to_wind_speed_dir(u_label, v_label)
            
            pred_converted[point_idx, 2] = speed_pred
            pred_converted[point_idx, 3] = dir_pred
            label_converted[point_idx, 2] = speed_label
            label_converted[point_idx, 3] = dir_label
            
            # Pressure: Pa to hPa
            pred_converted[point_idx, 4] = pressure_pred / 100.0
            label_converted[point_idx, 4] = pressure_label / 100.0
        
        converted_preds.append(pred_converted)
        converted_labels.append(label_converted)
    
    return converted_preds, converted_labels

def main(model, X_batch, y_batch, device):

    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 20

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        for i in range(len(X_batch)):
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)

                # Check for NaN in predictions
                has_nan = False
                for var in pred.surf_vars:
                    if torch.isnan(pred.surf_vars[var]).any():
                        print(f"NaN detected in pred.surf_vars[{var}] at batch {i}, skipping...")
                        has_nan = True
                        break
                
                for var in pred.atmos_vars:
                    if torch.isnan(pred.atmos_vars[var]).any():
                        print(f"NaN detected in pred.atmos_vars[{var}] at batch {i}, skipping...")
                        has_nan = True
                        break
                
                if has_nan:
                    continue

                # Zero the gradients
                optimizer.zero_grad()

                lat, lon = X.metadata.lat, X.metadata.lon

                pred = pred.to(device)
                y = y.to(device)

                loss = aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon)

                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {i}, skipping...")
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1

            except Exception as e:
                print(f"Error in batch {i}: {e}, skipping...")
                continue

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Valid batches: {valid_batches}/{len(X_batch)}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}]: No valid batches processed!")

    evaluation(model, X_batch, y_batch, device)

def evaluation(model, X_batch, y_batch, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for i in range(len(X_batch)):
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)

                # Check for NaN
                has_nan = False
                for var in pred.surf_vars:
                    if torch.isnan(pred.surf_vars[var]).any():
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"Skipping evaluation batch {i} due to NaN")
                    continue

                # 构建预测和真实值张量 - 包含所有11个变量（移除了tp）
                pred_tensor = torch.cat([
                    pred.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]),      # 0: temperature
                    pred.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]),    # 1: pm10
                    pred.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),    # 2: pm25
                    pred.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]),     # 3: so2
                    pred.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]),     # 4: no2
                    pred.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),      # 5: o3
                    pred.atmos_vars['q'][:,:,:,:,:].reshape([1, 1, 144]),    # 6: specific humidity
                    pred.surf_vars['10u'][:,:,:,:].reshape([1, 1, 144]),     # 7: u wind
                    pred.surf_vars['10v'][:,:,:,:].reshape([1, 1, 144]),     # 8: v wind
                    pred.surf_vars['msl'][:,:,:,:].reshape([1, 1, 144]),     # 9: pressure
                    pred.surf_vars['co'][:,:,:,:].reshape([1, 1, 144])       # 10: co
                ], dim=1)

                y_tensor = torch.cat([
                    y.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),
                    y.atmos_vars['q'][:,:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['10u'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['10v'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['msl'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['co'][:,:,:,:].reshape([1, 1, 144])
                ], dim=1)
            
                all_preds.append(pred_tensor.to(device))
                all_labels.append(y_tensor.to(device))

            except Exception as e:
                print(f"Error in evaluation batch {i}: {e}, skipping...")
                continue

    if len(all_preds) == 0:
        print("No valid predictions generated!")
        return

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    mse = F.mse_loss(all_preds, all_labels)

    # Save raw predictions
    all_preds_np = all_preds.cpu().detach().numpy()
    all_labels_np = all_labels.cpu().detach().numpy()
    np.save('all_preds_aurora_enhanced.npy', all_preds_np)
    np.save('all_labels_aurora_enhanced.npy', all_labels_np)

    print(f'Mean Squared Error: {mse.item()}')

    # Create enhanced visualization with raw Aurora outputs
    plot_predictions_vs_labels_enhanced(all_preds, all_labels, 'enhanced_aurora_predictions')

    # Convert to physical units for meteorological visualization
    converted_preds, converted_labels = convert_predictions_to_physical_units(all_preds, all_labels)
    
    # Save converted predictions
    np.save('converted_preds_weather.npy', np.array(converted_preds))
    np.save('converted_labels_weather.npy', np.array(converted_labels))
    
    # Create meteorological visualization
    create_weather_visualization(all_preds, all_labels, converted_preds, converted_labels)

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

    # Physical loss: could add divergence-free constraint for wind
    div_free_loss = 0  # Simplified for now

    # Total loss
    total_loss = total_surf_loss + total_atmos_loss + reg_weight_div * div_free_loss
    return total_loss

if __name__ == '__main__':

    torch.set_num_threads(4)

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    df = process_excel2(filename_weather, filename_airquality)

    X_batch, y_batch = form_aurora_batch(df)

    # Enhanced Aurora model with meteorological variables (without tp)
    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "10u", "10v", "msl", "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        atmos_vars=("t", "u", "v", "q"),
        autocast=True,
    )

    # Calculate mean and std for air quality variables only
    mean_values = df.select_dtypes(include=['number']).mean()
    std_values = df.select_dtypes(include=['number']).std()

    # Set normalisation for air quality variables only
    # Let Aurora handle meteorological variables with its defaults
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

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Attempting evaluation with current model state...")
        evaluation(model, X_batch, y_batch, device)
