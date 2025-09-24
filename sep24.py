# train
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

def create_october_filtered_visualization_fixed(all_preds, all_labels, X_batch, y_batch, df_original):
    """
    修复版本：从完整的预测结果中过滤10月份数据进行可视化
    """
    print("开始过滤10月份数据用于可视化（修复版）...")
    
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
    
    # 调试时间逻辑
    debug_batch_time_logic(X_batch, y_batch, df_original)
    
    # 创建可视化
    create_station_grouped_visualization(october_preds, october_labels, october_timestamps)
    create_detailed_temperature_plot(october_preds, october_labels, october_timestamps)
    create_individual_station_plots(october_preds, october_labels, october_timestamps)

    # 使用修复版的导出函数
    export_shangqingsi_predictions_to_excel_fixed(october_preds, october_labels, october_timestamps, 
                                                   [X_batch[i] for i in october_indices], 
                                                   [y_batch[i] for i in october_indices if i < len(y_batch)], 
                                                   df_original)
    
    return october_preds, october_labels, october_timestamps

def main_fixed(model, X_batch, y_batch, device, df_original):
    """
    修复版主函数：保持原来的完整训练+evaluation流程，但修复导出功能
    """
    # 训练阶段（保持不变）
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

    # Evaluation阶段（保持不变）
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

    # 创建10月份过滤的可视化（修复版）
    october_preds, october_labels, october_timestamps = create_october_filtered_visualization_fixed(
        all_preds, all_labels, X_batch, y_batch, df_original)
    
    # 保存10月份数据
    np.save('october_preds_filtered_fixed.npy', october_preds.cpu().detach().numpy())
    np.save('october_labels_filtered_fixed.npy', october_labels.cpu().detach().numpy())
    
    # 保存时间戳
    times_df = pd.DataFrame({'timestamp': october_timestamps})
    times_df.to_csv('october_timestamps_fixed.csv', index=False)
    
    print(f"\n处理完成（修复版）:")
    print(f"- 总数据形状: {all_preds.shape}")
    print(f"- 10月份数据形状: {october_preds.shape}")
    print(f"- 生成的可视化图表基于10月份数据")
    print(f"- 已生成两个版本的Excel文件：预处理版和原始数据版")

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
    station_names = ['Shangqingsi', 'Tangjiabei', 'Tiansheng', 'Longjingwan']
    
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
    station_names = ['Shangqingsi', 'Tangjiabei', 'Tiansheng', 'Longjingwan']
    
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
    station_names = ['Shangqingsi', 'Tangjiabei', 'Tiansheng', 'Longjingwan']
    
    # 选择要显示的变量（每个站点图显示9个主要变量）
    display_vars = ['Temperature (K)', 'U Wind (m/s)', 'V Wind (m/s)',
                    'Pressure (Pa)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)', 
                    'SO2 (μg/m³)', 'NO2 (μg/m³)', 'O3 (μg/m³)',
                    'CO (mg/m³)', 'Humidity (kg/kg)']
    
    for station_idx, (start, end) in enumerate(station_ranges):
        station_name = station_names[station_idx]
        
        # 4x3的子图
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
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


def export_shangqingsi_predictions_to_excel_fixed(preds, labels, timestamps, X_batch, y_batch, df_original):
    """
    修复版本：导出上清寺站点的10月份预测数据到Excel
    创建两个版本：预处理数据版和原始数据版
    """
    print("=== 开始导出修复版上清寺预测数据 ===")
    
    # 变量定义（与构建batch时保持一致）
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
    
    shangqingsi_range = (0, 36)  # 上清寺是第1个站点，索引0
    station_name = "上清寺"
    
    # 准备原始数据字典用于对比
    df_shangqingsi_original = df_original[df_original['station_name'] == '上清寺'].copy()
    df_shangqingsi_original['time'] = pd.to_datetime(df_shangqingsi_original['time'])
    df_shangqingsi_original = df_shangqingsi_original.sort_values('time').reset_index(drop=True)
    
    # 创建时间到原始数据的映射
    original_data_map = {}
    for _, row in df_shangqingsi_original.iterrows():
        time_key = row['time']
        original_data_map[time_key] = {
            'temperature_raw': row['temperature'],
            'humidity_raw': row['humidity'], 
            'pressure_raw': row['pressure'],
            'wind_speed_raw': row['wind_speed'],
            'wind_direction_raw': row['wind_direction'],
            'pm25_raw': row['pm25'],
            'pm10_raw': row['pm10'],
            'so2_raw': row['so2'],
            'no2_raw': row['no2'],
            'o3_raw': row['o3'],
            'co_raw': row['co']
        }
    
    export_data_processed = []
    export_data_original = []
    
    print(f"处理 {len(preds)} 个批次的数据...")
    
    for batch_idx in range(len(preds)):
        # 修复：正确获取输入时间和预测时间
        input_time = X_batch[batch_idx].metadata.time[0]  # 输入数据的时间
        
        # 修复：使用对应的y_batch获取真实的预测时间
        if batch_idx < len(y_batch):
            pred_time = y_batch[batch_idx].metadata.time[0]  # 预测的目标时间
        else:
            # 如果y_batch长度不够，手动计算预测时间（应该是input_time + 3小时）
            pred_time = input_time + pd.Timedelta(hours=3)
        
        print(f"Batch {batch_idx}: 输入时间 {input_time} -> 预测时间 {pred_time}")
        
        # === 版本1：预处理数据版本 ===
        row_data_processed = {
            '站点名称': station_name,
            '输入时间': input_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测时间': pred_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测间隔(小时)': int((pred_time - input_time).total_seconds() / 3600)
        }
        
        # 添加预处理后的预测值和真实值
        start, end = shangqingsi_range
        for var_name, var_idx in variables.items():
            # 使用平均值（如原代码）
            pred_mean = preds[batch_idx, var_idx, start:end].mean().cpu().numpy()
            true_mean = labels[batch_idx, var_idx, start:end].mean().cpu().numpy()
            
            clean_var_name = var_name.replace(' ', '_').replace('(', '').replace(')', '').replace('μ', 'u').replace('³', '3')
            
            row_data_processed[f'{clean_var_name}_pred'] = float(pred_mean)
            row_data_processed[f'{clean_var_name}_true_processed'] = float(true_mean)
            row_data_processed[f'{clean_var_name}_error'] = abs(float(pred_mean - true_mean))
        
        export_data_processed.append(row_data_processed)
        
        # === 版本2：原始数据对比版本 ===
        row_data_original = {
            '站点名称': station_name,
            '输入时间': input_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测时间': pred_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测间隔(小时)': int((pred_time - input_time).total_seconds() / 3600)
        }
        
        # 获取原始数据（如果存在）
        original_input = original_data_map.get(input_time, {})
        original_pred = original_data_map.get(pred_time, {})
        
        # 预测值（来自模型，需要转换回原始单位）
        pred_temp_processed = preds[batch_idx, 0, start:end].mean().cpu().numpy()  # 温度(K)
        pred_pressure_processed = preds[batch_idx, 3, start:end].mean().cpu().numpy()  # 气压(Pa) 
        pred_humidity_processed = preds[batch_idx, 10, start:end].mean().cpu().numpy()  # 比湿度
        pred_u_wind = preds[batch_idx, 1, start:end].mean().cpu().numpy()  # u分量
        pred_v_wind = preds[batch_idx, 2, start:end].mean().cpu().numpy()  # v分量
        
        # 转换预测值回原始格式
        from utils import uv_to_wind_speed_dir, specific_humidity_to_rh
        
        pred_wind_speed, pred_wind_dir = uv_to_wind_speed_dir(pred_u_wind, pred_v_wind)
        pred_humidity_percent = specific_humidity_to_rh(pred_humidity_processed, pred_temp_processed, pred_pressure_processed)
        
        # 原始格式的预测值
        row_data_original.update({
            # 输入时间的原始数据
            'input_temp_raw': original_input.get('temperature_raw', np.nan),
            'input_humidity_raw': original_input.get('humidity_raw', np.nan),
            'input_pressure_raw': original_input.get('pressure_raw', np.nan),
            'input_wind_speed_raw': original_input.get('wind_speed_raw', np.nan),
            'input_wind_dir_raw': original_input.get('wind_direction_raw', np.nan),
            'input_pm25_raw': original_input.get('pm25_raw', np.nan),
            'input_pm10_raw': original_input.get('pm10_raw', np.nan),
            
            # 预测时间的真实原始数据
            'true_temp_raw': original_pred.get('temperature_raw', np.nan),
            'true_humidity_raw': original_pred.get('humidity_raw', np.nan),
            'true_pressure_raw': original_pred.get('pressure_raw', np.nan),
            'true_wind_speed_raw': original_pred.get('wind_speed_raw', np.nan),
            'true_wind_dir_raw': original_pred.get('wind_direction_raw', np.nan),
            'true_pm25_raw': original_pred.get('pm25_raw', np.nan),
            'true_pm10_raw': original_pred.get('pm10_raw', np.nan),
            'true_so2_raw': original_pred.get('so2_raw', np.nan),
            'true_no2_raw': original_pred.get('no2_raw', np.nan),
            'true_o3_raw': original_pred.get('o3_raw', np.nan),
            'true_co_raw': original_pred.get('co_raw', np.nan),
            
            # 模型预测值（转换回原始格式）
            'pred_temp_raw': float(pred_temp_processed),  # K单位保持不变
            'pred_humidity_raw': float(pred_humidity_percent),  # 转换回百分比
            'pred_pressure_raw': float(pred_pressure_processed),  # Pa单位保持
            'pred_wind_speed_raw': float(pred_wind_speed),
            'pred_wind_dir_raw': float(pred_wind_dir),
            'pred_pm25_raw': float(preds[batch_idx, 5, start:end].mean().cpu().numpy()),
            'pred_pm10_raw': float(preds[batch_idx, 4, start:end].mean().cpu().numpy()),
            'pred_so2_raw': float(preds[batch_idx, 6, start:end].mean().cpu().numpy()),
            'pred_no2_raw': float(preds[batch_idx, 7, start:end].mean().cpu().numpy()),
            'pred_o3_raw': float(preds[batch_idx, 8, start:end].mean().cpu().numpy()),
            'pred_co_raw': float(preds[batch_idx, 9, start:end].mean().cpu().numpy()),
        })
        
        # 计算误差（原始格式）
        if not np.isnan(original_pred.get('temperature_raw', np.nan)):
            row_data_original['temp_error_raw'] = abs(float(pred_temp_processed) - original_pred['temperature_raw'])
        if not np.isnan(original_pred.get('pm25_raw', np.nan)):
            row_data_original['pm25_error_raw'] = abs(float(preds[batch_idx, 5, start:end].mean().cpu().numpy()) - original_pred['pm25_raw'])
            
        export_data_original.append(row_data_original)
    
    # 转换为DataFrame并保存
    df_export_processed = pd.DataFrame(export_data_processed)
    df_export_original = pd.DataFrame(export_data_original)
    
    # 按预测时间排序
    df_export_processed = df_export_processed.sort_values('预测时间')
    df_export_original = df_export_original.sort_values('预测时间')
    
    # 保存到Excel
    filename_processed = '上清寺_10月份预测结果_预处理版.xlsx'
    filename_original = '上清寺_10月份预测结果_原始数据版.xlsx'
    
    df_export_processed.to_excel(filename_processed, index=False, engine='openpyxl')
    df_export_original.to_excel(filename_original, index=False, engine='openpyxl')
    
    print(f"\n=== 导出完成 ===")
    print(f"预处理版数据已导出到: {filename_processed}")
    print(f"原始数据版已导出到: {filename_original}")
    print(f"包含 {len(df_export_processed)} 条记录")
    
    # 验证时间逻辑
    time_diff_hours = []
    for _, row in df_export_processed.iterrows():
        input_time = pd.to_datetime(row['输入时间'])
        pred_time = pd.to_datetime(row['预测时间'])
        diff_hours = (pred_time - input_time).total_seconds() / 3600
        time_diff_hours.append(diff_hours)
    
    print(f"时间间隔验证: 最小{min(time_diff_hours)}小时, 最大{max(time_diff_hours)}小时, 平均{np.mean(time_diff_hours):.1f}小时")
    
    return df_export_processed, df_export_original

def debug_batch_time_logic(X_batch, y_batch, df_original):
    """
    调试batch时间逻辑，验证预测时间计算是否正确
    """
    print("=== 调试Batch时间逻辑 ===")
    
    # 检查前5个batch的时间逻辑
    for i in range(min(5, len(X_batch), len(y_batch))):
        input_time = X_batch[i].metadata.time[0]
        pred_time = y_batch[i].metadata.time[0]
        time_diff = (pred_time - input_time).total_seconds() / 3600
        
        print(f"Batch {i}:")
        print(f"  输入时间: {input_time}")
        print(f"  预测时间: {pred_time}") 
        print(f"  时间差: {time_diff:.1f} 小时")
        
        # 检查原始数据中是否存在这些时间点
        df_shangqingsi = df_original[df_original['station_name'] == '上清寺']
        df_shangqingsi['time'] = pd.to_datetime(df_shangqingsi['time'])
        
        input_exists = input_time in df_shangqingsi['time'].values
        pred_exists = pred_time in df_shangqingsi['time'].values
        
        print(f"  输入时间在原始数据中: {'存在' if input_exists else '不存在'}")
        print(f"  预测时间在原始数据中: {'存在' if pred_exists else '不存在'}")
        print()
    
    print("=== 检查原始数据时间序列 ===")
    df_shangqingsi = df_original[df_original['station_name'] == '上清寺'].copy()
    df_shangqingsi['time'] = pd.to_datetime(df_shangqingsi['time'])
    df_shangqingsi = df_shangqingsi.sort_values('time')
    
    print(f"上清寺原始数据时间范围: {df_shangqingsi['time'].min()} 到 {df_shangqingsi['time'].max()}")
    print(f"总计 {len(df_shangqingsi)} 条记录")
    
    # 检查时间间隔
    time_diffs = df_shangqingsi['time'].diff().dropna()
    unique_intervals = time_diffs.unique()
    print(f"时间间隔: {[str(interval) for interval in unique_intervals]}")
    
    return True

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

    # 保存原始数据用于对比
    print("=== 读取原始数据用于对比 ===")
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.abspath(os.path.join(base_path, filename_weather))
    airquality_file_path = os.path.abspath(os.path.join(base_path, filename_airquality))
    
    # 读取原始气象数据
    df_weather_raw = pd.read_excel(io=weather_file_path, header=0)
    df_weather_raw = df_weather_raw[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度', 
                           '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather_raw.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed', 'wind_direction']
    
    # 读取原始空气质量数据
    df_airquality_raw = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0)
    df_airquality_raw = pd.concat(df_airquality_raw.values(), ignore_index=True)
    air_columns = ['station_name', 'monitoring_time', 'longitude', 'latitude', 
                  'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    df_airquality_raw = df_airquality_raw[air_columns].copy()
    df_airquality_raw.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    # 合并原始数据
    df_airquality_raw['time'] = pd.to_datetime(df_airquality_raw['time'])
    df_weather_raw['time'] = pd.to_datetime(df_weather_raw['time'])
    df_original = pd.merge(df_weather_raw, df_airquality_raw, on=['station_name', 'time'], how='inner')
    
    print(f"原始数据形状: {df_original.shape}")
    print(f"原始数据站点: {df_original['station_name'].unique()}")

    # 调用现有的数据处理函数
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
        # 使用修复版的主函数，传入原始数据
        main_fixed(model, X_batch, y_batch, device, df_original)
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
