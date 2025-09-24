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
    
    # 创建四个单独的站点详细图
    create_individual_station_plots(october_preds, october_labels, october_timestamps)
    
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
    station_names = ['Tiansheng', 'Longjingwan', 'Tangjiabei', 'Shangqingsi']
    
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
        ax.plot(time_indices, pred_values, '--', color='#FF6B6B', linewidth=2, alpha=0.8, label='Predicted')
        ax.plot(time_indices, true_values, '-', color='#66BB6A', linewidth=2, alpha=0.9, label='True')
        
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

def create_individual_station_plots(preds, labels, timestamps):
    """
    为每个站点创建单独的详细图表，显示所有变量，并在左上角标注RMSE
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
    
    station_ranges = [(0, 36), (36, 72), (72, 108), (108, 144)]
    station_names = ['Tiansheng', 'Longjingwan', 'Tangjiabei', 'Yuxinjie']
    
    # 选择要显示的变量（每个站点图显示9个主要变量）
    display_vars = ['Temperature (K)', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 
                   'Pressure (Pa)', 'NO2 (μg/m³)', 'O3 (μg/m³)',
                   'SO2 (μg/m³)', 'CO (mg/m³)', 'Humidity (kg/kg)']
    
    for station_idx, (start, end) in enumerate(station_ranges):
        station_name = station_names[station_idx]
        
        # 为每个站点创建3x3的子图
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'{station_name} Station - All Variables Comparison (October)', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        station_rmse_total = 0.0
        
        for var_idx, var_name in enumerate(display_vars):
            ax = axes_flat[var_idx]
            var_data_idx = variables[var_name]
            
            pred_values = []
            true_values = []
            
            # 提取该站点该变量的所有时间点数据
            for batch_idx in range(len(preds)):
                pred_mean = preds[batch_idx, var_data_idx, start:end].mean().cpu().numpy()
                true_mean = labels[batch_idx, var_data_idx, start:end].mean().cpu().numpy()
                
                pred_values.append(pred_mean)
                true_values.append(true_mean)
            
            # 计算该变量的RMSE
            pred_array = np.array(pred_values)
            true_array = np.array(true_values)
            var_rmse = np.sqrt(np.mean((pred_array - true_array) ** 2))
            station_rmse_total += var_rmse
            
            # 绘制预测vs真实值 - 使用浅红绿配色
            time_indices = range(len(pred_values))
            ax.plot(time_indices, pred_values, '--', color='#FF6B6B', linewidth=1.5, alpha=0.8, label='Predicted')
            ax.plot(time_indices, true_values, '-', color='#66BB6A', linewidth=1.5, alpha=0.9, label='True')
            
            # 设置标题和标签
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Index (October)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # 只在第一个子图显示图例
            if var_idx == 0:
                ax.legend(fontsize=8)
            
            # 在每个子图右下角显示该变量的RMSE - 使用浅绿色
            ax.text(0.95, 0.05, f'RMSE: {var_rmse:.3f}', transform=ax.transAxes, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E8', alpha=0.8),
                   fontsize=8)
            
            # 计算相关系数 - 使用浅红色
            if len(pred_values) > 1:
                correlation = np.corrcoef(pred_values, true_values)[0, 1]
                ax.text(0.05, 0.05, f'r = {correlation:.3f}', transform=ax.transAxes, 
                       verticalalignment='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE8E8', alpha=0.8),
                       fontsize=8)
        
        # 计算整个站点的平均RMSE
        avg_rmse = station_rmse_total / len(display_vars)
        
        # 在整个图的左上角显示站点总体RMSE - 使用浅灰蓝色
        fig.text(0.02, 0.98, f'Station RMSE: {avg_rmse:.3f}', 
                transform=fig.transFigure, fontsize=14, fontweight='bold',
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(f'{station_name}_detailed_october.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"生成图表: {station_name}_detailed_october.png (平均RMSE: {avg_rmse:.3f})")
    
    print(f"完成所有单独站点图表生成！")

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
    num_epochs = 1

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

def debug_data_merge(filename_weather, filename_airquality):
    """
    调试数据合并过程，找出上清寺数据丢失的原因
    """
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.abspath(os.path.join(base_path, filename_weather))
    airquality_file_path = os.path.abspath(os.path.join(base_path, filename_airquality))

    print("=== 调试数据合并过程 ===")
    
    # 检查气象数据
    df_weather = pd.read_excel(io=weather_file_path, header=0)
    print(f"气象数据原始形状: {df_weather.shape}")
    print(f"气象数据站点: {df_weather['station_name'].unique()}")
    
    weather_station_counts = df_weather['station_name'].value_counts()
    print("气象数据各站点记录数:")
    for station, count in weather_station_counts.items():
        print(f"  {station}: {count}")
    
    # 检查空气质量数据
    df_airquality = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0)
    print(f"\n空气质量数据sheet数量: {len(df_airquality)}")
    print(f"Sheet名称: {list(df_airquality.keys())}")
    
    # 检查每个sheet的列名
    for sheet_name, sheet_df in df_airquality.items():
        print(f"\n{sheet_name} sheet列名: {sheet_df.columns.tolist()}")
        if 'monitoring_time' in sheet_df.columns:
            print(f"{sheet_name} 有 monitoring_time 列")
        else:
            print(f"{sheet_name} 没有 monitoring_time 列，可能需要检查时间列名")
    
    # 合并所有sheet - 先处理时间列问题
    air_data_list = []
    for sheet_name, sheet_df in df_airquality.items():
        sheet_copy = sheet_df.copy()
        
        # 如果没有monitoring_time列，查找其他时间列
        time_columns = [col for col in sheet_copy.columns if 'time' in col.lower() or '时间' in col]
        if 'monitoring_time' not in sheet_copy.columns and time_columns:
            print(f"{sheet_name}: 使用 {time_columns[0]} 作为时间列")
            sheet_copy.rename(columns={time_columns[0]: 'monitoring_time'}, inplace=True)
        
        air_data_list.append(sheet_copy)
    
    df_air_combined = pd.concat(air_data_list, ignore_index=True)
    print(f"合并后空气质量数据形状: {df_air_combined.shape}")
    print(f"合并后列名: {df_air_combined.columns.tolist()}")
    
    if 'station_name' in df_air_combined.columns:
        print(f"空气质量数据站点: {df_air_combined['station_name'].unique()}")
        air_station_counts = df_air_combined['station_name'].value_counts()
        print("空气质量数据各站点记录数:")
        for station, count in air_station_counts.items():
            print(f"  {station}: {count}")
    
    # 修复时间列处理
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # 重命名时间列并处理
    if 'monitoring_time' in df_air_combined.columns:
        df_air_combined.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    try:
        df_air_combined['time'] = pd.to_datetime(df_air_combined['time'])
        print("空气质量数据时间转换成功")
    except Exception as e:
        print(f"空气质量数据时间转换失败: {e}")
        # 检查时间列的样本值
        print(f"时间列样本: {df_air_combined['time'].head()}")
        return None
    
    print(f"\n气象数据时间范围: {df_weather['time'].min()} 到 {df_weather['time'].max()}")
    print(f"空气质量数据时间范围: {df_air_combined['time'].min()} 到 {df_air_combined['time'].max()}")
    
    # 检查上清寺数据
    print("\n=== 检查上清寺数据 ===")
    shangqingsi_weather = df_weather[df_weather['station_name'] == '上清寺']
    shangqingsi_air = df_air_combined[df_air_combined['station_name'] == '上清寺']
    
    print(f"上清寺气象数据: {len(shangqingsi_weather)} 条记录")
    if len(shangqingsi_weather) > 0:
        print(f"  时间范围: {shangqingsi_weather['time'].min()} 到 {shangqingsi_weather['time'].max()}")
    
    print(f"上清寺空气质量数据: {len(shangqingsi_air)} 条记录")
    if len(shangqingsi_air) > 0:
        print(f"  时间范围: {shangqingsi_air['time'].min()} 到 {shangqingsi_air['time'].max()}")
    
    # 检查合并后结果
    print("\n=== 检查合并结果 ===")
    try:
        merged_inner = pd.merge(df_weather, df_air_combined, on=['station_name', 'time'], how='inner')
        merged_left = pd.merge(df_weather, df_air_combined, on=['station_name', 'time'], how='left')
        
        print(f"Inner join结果: {merged_inner.shape}")
        print(f"Inner join站点: {merged_inner['station_name'].unique()}")
        
        print(f"Left join结果: {merged_left.shape}")
        print(f"Left join站点: {merged_left['station_name'].unique()}")
        
        # 检查每个站点的合并结果
        print("\nInner join各站点记录数:")
        inner_counts = merged_inner['station_name'].value_counts()
        for station, count in inner_counts.items():
            print(f"  {station}: {count}")
            
        return merged_inner
        
    except Exception as e:
        print(f"数据合并失败: {e}")
        return None
    
    return merged_inner

if __name__ == '__main__':
    torch.set_num_threads(4)

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    debug_result = debug_data_merge(filename_weather, filename_airquality)

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
