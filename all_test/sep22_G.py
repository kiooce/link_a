import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# 必要的导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from pathlib import Path

# Aurora相关导入
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

# 导入你的工具函数
from utils import process_excel2, form_aurora_batch

def create_october_filtered_visualization(all_preds, all_labels, X_batch, y_batch):
    """
    从完整的预测结果中过滤10月份数据进行可视化
    """
    print("开始过滤10月份数据用于可视化...")
    
    # 过滤10月份的索引
    october_indices = []
    october_timestamps = []
    
    for i, X in enumerate(X_batch):
        if i < len(all_preds):  # 确保索引有效
            batch_time = X.metadata.time[0]
            if batch_time.month == 10:
                october_indices.append(i)
                october_timestamps.append(batch_time)
    
    print(f"找到 {len(october_indices)} 个10月份批次进行可视化")
    
    if len(october_indices) == 0:
        print("没有找到10月份数据，使用所有数据进行可视化")
        october_indices = list(range(len(all_preds)))
        october_timestamps = [f"Batch_{i}" for i in october_indices]
    
    # 提取10月份的预测和标签数据
    october_preds = all_preds[october_indices]
    october_labels = all_labels[october_indices]
    
    print(f"10月份数据形状: {october_preds.shape}")
    
    # 创建可视化
    create_station_grouped_visualization(october_preds, october_labels, october_timestamps)
    create_detailed_temperature_plot(october_preds, october_labels, october_timestamps)
    
    return october_preds, october_labels, october_timestamps

def create_station_grouped_visualization(preds, labels, timestamps):
    """
    创建按站点分组的多变量可视化
    """
    # 变量定义
    variables = {
        'Temperature (K)': 0,
        'U Wind (m/s)': 1, 
        'V Wind (m/s)': 2,
        'Pressure (Pa)': 3,
        'PM10 (μg/m³)': 4,
        'PM2.5 (μg/m³)': 5,
        'SO2 (μg/m³)': 6,
        'NO2 (μg/m³)': 7,
        'O3 (μg/m³)': 8,
        'CO (mg/m³)': 9,
        'Humidity (kg/kg)': 10
    }
    
    # 站点设置
    station_ranges = [(0, 36), (36, 72), (72, 108), (108, 144)]
    station_names = ['Tiansheng', 'Longjingwan', 'Tangjiabei', 'Yuxinjie']
    
    # 创建图表
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Aurora Model Predictions vs True Values - October Data by Station', 
                 fontsize=16, fontweight='bold')
    
    # 选择重要变量
    important_vars = ['Temperature (K)', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 
                     'Pressure (Pa)', 'NO2 (μg/m³)', 'O3 (μg/m³)',
                     'SO2 (μg/m³)', 'CO (mg/m³)', 'U Wind (m/s)',
                     'V Wind (m/s)', 'Humidity (kg/kg)']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, var_name in enumerate(important_vars):
        if idx >= 12:
            break
            
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        var_idx = variables[var_name]
        
        # 为每个站点绘制数据
        for station_idx, (start, end) in enumerate(station_ranges):
            pred_values = []
            true_values = []
            
            for batch_idx in range(len(preds)):
                # 取该站点区域的平均值
                pred_mean = preds[batch_idx, var_idx, start:end].mean().cpu().numpy()
                true_mean = labels[batch_idx, var_idx, start:end].mean().cpu().numpy()
                
                pred_values.append(pred_mean)
                true_values.append(true_mean)
            
            # 绘制预测值和真实值
            time_indices = range(len(pred_values))
            ax.plot(time_indices, pred_values, '--', color=colors[station_idx], 
                   linewidth=1, alpha=0.8, label=f'{station_names[station_idx]} Pred')
            ax.plot(time_indices, true_values, '-', color=colors[station_idx], 
                   linewidth=1.5, alpha=0.9, label=f'{station_names[station_idx]} True')
        
        ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index (October)', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
        
        ax.tick_params(labelsize=8)
    
    # 移除多余子图
    for idx in range(len(important_vars), 12):
        row = idx // 3
        col = idx % 3
        axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig('october_predictions_by_station.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成图表: october_predictions_by_station.png")

def create_detailed_temperature_plot(preds, labels, timestamps):
    """
    创建温度的详细对比图
    """
    station_ranges = [(0, 36), (36, 72), (72, 108), (108, 144)]
    station_names = ['Tiansheng', 'Longjingwan', 'Tangjiabei', 'Yuxinjie']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temperature - Detailed Comparison by Station (October)', 
                 fontsize=14, fontweight='bold')
    
    axes_flat = axes.flatten()
    var_idx = 0  # 温度是第0个变量
    
    for station_idx, (start, end) in enumerate(station_ranges):
        ax = axes_flat[station_idx]
        
        pred_values = []
        true_values = []
        
        for batch_idx in range(len(preds)):
            pred_mean = preds[batch_idx, var_idx, start:end].mean().cpu().numpy()
            true_mean = labels[batch_idx, var_idx, start:end].mean().cpu().numpy()
            
            pred_values.append(pred_mean)
            true_values.append(true_mean)
        
        time_indices = range(len(pred_values))
        ax.plot(time_indices, pred_values, 'r--', linewidth=2, alpha=0.7, label='Predicted')
        ax.plot(time_indices, true_values, 'b-', linewidth=2, alpha=0.8, label='True')
        
        ax.set_title(f'{station_names[station_idx]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Index (October)', fontsize=10)
        ax.set_ylabel('Temperature (K)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 计算相关系数
        if len(pred_values) > 1:
            correlation = np.corrcoef(pred_values, true_values)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('temperature_detailed_october.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成图表: temperature_detailed_october.png")

def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    """Aurora损失函数"""
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

def main(model, X_batch, y_batch, device):
    """
    保持原来的完整训练+evaluation流程
    """
    # 训练阶段
    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 30

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        for i in range(len(X_batch)):
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)

                has_nan = False
                for var in pred.surf_vars:
                    if torch.isnan(pred.surf_vars[var]).any():
                        print(f"NaN detected in pred.surf_vars[{var}] at batch {i}, skipping...")
                        has_nan = True
                        break
                
                if has_nan:
                    continue

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

    # Evaluation阶段（所有数据）
    print("开始evaluation所有数据...")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for i in range(len(X_batch)):
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)

                has_nan = False
                for var in pred.surf_vars:
                    if torch.isnan(pred.surf_vars[var]).any():
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"Skipping evaluation batch {i} due to NaN")
                    continue

                # 构建张量（保持原来的顺序）
                pred_tensor = torch.cat([
                    pred.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['10u'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['10v'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['msl'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),
                    pred.surf_vars['co'][:,:,:,:].reshape([1, 1, 144]),
                    pred.atmos_vars['q'][:,:,:,:,:].reshape([1, 1, 144]),
                ], dim=1)

                y_tensor = torch.cat([
                    y.surf_vars['2t'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['10u'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['10v'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['msl'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['pm10'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['pm25'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['so2'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['no2'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['o3'][:,:,:,:].reshape([1, 1, 144]),
                    y.surf_vars['co'][:,:,:,:].reshape([1, 1, 144]),
                    y.atmos_vars['q'][:,:,:,:,:].reshape([1, 1, 144])
                ], dim=1)
            
                all_preds.append(pred_tensor.to(device))
                all_labels.append(y_tensor.to(device))

            except Exception as e:
                print(f"Error in evaluation batch {i}: {e}, skipping...")
                continue

    if len(all_preds) == 0:
        print("No valid predictions generated!")
        return

    # 合并所有预测结果
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    mse = F.mse_loss(all_preds, all_labels)
    print(f'Mean Squared Error: {mse.item()}')

    # 保存完整数据
    all_preds_np = all_preds.cpu().detach().numpy()
    all_labels_np = all_labels.cpu().detach().numpy()
    np.save('all_preds_aurora_complete.npy', all_preds_np)
    np.save('all_labels_aurora_complete.npy', all_labels_np)

    # 创建10月份过滤的可视化
    october_preds, october_labels, october_timestamps = create_october_filtered_visualization(
        all_preds, all_labels, X_batch, y_batch)
    
    # 保存10月份数据
    np.save('october_preds_filtered.npy', october_preds.cpu().detach().numpy())
    np.save('october_labels_filtered.npy', october_labels.cpu().detach().numpy())
    
    # 保存时间戳
    times_df = pd.DataFrame({'timestamp': october_timestamps})
    times_df.to_csv('october_timestamps.csv', index=False)
    
    print(f"\n处理完成:")
    print(f"- 总数据形状: {all_preds.shape}")
    print(f"- 10月份数据形状: {october_preds.shape}")
    print(f"- 生成的可视化图表基于10月份数据")

if __name__ == '__main__':
    torch.set_num_threads(4)

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    df = process_excel2(filename_weather, filename_airquality)
    X_batch, y_batch = form_aurora_batch(df)

    # Enhanced Aurora model
    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "10u", "10v", "msl", "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        atmos_vars=("t", "u", "v", "q"),
        autocast=True,
    )

    # 设置归一化参数
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

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
