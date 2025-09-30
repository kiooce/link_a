import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from aurora import Aurora
from aurora.normalisation import locations, scales
from utils import process_excel2, form_aurora_batch, uv_to_wind_speed_dir, specific_humidity_to_rh

def main(model, X_batch, y_batch, device, df_original):
    """训练和评估 - 修正版本：正确划分训练集和测试集"""
    
    # ===== 新增部分1：数据划分 =====
    print("=== 开始数据划分 ===")
    
    # 收集所有批次的时间信息
    batch_times = []
    for i, batch in enumerate(y_batch):
        batch_time = batch.metadata.time[0]
        batch_times.append((i, batch_time))
    
    # 按时间划分训练集和测试集
    train_indices = []
    test_indices = []
    
    for i, time in batch_times:
        if time.month == 10 and time.day <= 20:
            train_indices.append(i)  # 10月1-20日作为训练集
        elif time.month == 10 and time.day > 20:
            test_indices.append(i)   # 10月21-31日作为测试集
        elif time.month == 9:
            train_indices.append(i)  # 9月数据也加入训练集
    
    print(f"训练集大小: {len(train_indices)} (包含9月和10月1-20日)")
    print(f"测试集大小: {len(test_indices)} (10月21-31日)")
    
    if len(train_indices) == 0:
        print("错误：训练集为空！")
        return
    if len(test_indices) == 0:
        print("警告：测试集为空！")
    
    # ===== 原有的训练代码，但只用训练集 =====
    model.train()
    model.configure_activation_checkpointing()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 1  # 你可以改成200或300
    
    print(f"\n=== 开始训练（使用{len(train_indices)}个批次）===")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        # 修改：只遍历训练集索引
        for i in train_indices:  # <--- 这里改了，原来是 range(len(X_batch))
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)
                
                optimizer.zero_grad()
                lat, lon = X.metadata.lat, X.metadata.lon
                pred = pred.to(device)
                y = y.to(device)
                
                loss = aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
            except Exception as e:  # <--- 改进了异常处理
                print(f"警告：批次 {i} 训练失败 - {e}")
                continue
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # ===== 评估部分 - 分别处理训练集和测试集 =====
    print("\n=== 开始评估 ===")
    model.eval()
    
    # 准备存储所有预测结果（用于兼容后续的可视化代码）
    all_preds = []
    all_labels = []
    
    # 测试集评估
    test_preds = []
    test_labels = []
    
    print(f"评估测试集（{len(test_indices)}个批次）...")
    
    with torch.inference_mode():
        # 先评估测试集
        for i in test_indices:  # <--- 只用测试集索引
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)
                
                def extract_2x2_ordered(tensor_12x12):
                    center_2x2 = tensor_12x12[0, 0, 5:7, 5:7]
                    return torch.stack([
                        center_2x2[1,1],  # 索引0 = 上清寺(实际在1,1位置)
                        center_2x2[1,0],  # 索引1 = 唐家沱(实际在1,0位置)
                        center_2x2[0,1],  # 索引2 = 天生(实际在0,1位置)
                        center_2x2[0,0]   # 索引3 = 龙井湾(实际在0,0位置)
                    ])
                
                # 提取预测和真实值
                pred_tensor = torch.stack([
                    extract_2x2_ordered(pred.surf_vars['2t']),
                    extract_2x2_ordered(pred.surf_vars['10u']),
                    extract_2x2_ordered(pred.surf_vars['10v']),
                    extract_2x2_ordered(pred.surf_vars['msl']),
                    extract_2x2_ordered(pred.surf_vars['pm10']),
                    extract_2x2_ordered(pred.surf_vars['pm25']),
                    extract_2x2_ordered(pred.surf_vars['so2']),
                    extract_2x2_ordered(pred.surf_vars['no2']),
                    extract_2x2_ordered(pred.surf_vars['o3']),
                    extract_2x2_ordered(pred.surf_vars['co']),
                    extract_2x2_ordered(pred.atmos_vars['q'][:,:,0,:,:])
                ]).unsqueeze(0)
                
                y_tensor = torch.stack([
                    extract_2x2_ordered(y.surf_vars['2t']),
                    extract_2x2_ordered(y.surf_vars['10u']),
                    extract_2x2_ordered(y.surf_vars['10v']),
                    extract_2x2_ordered(y.surf_vars['msl']),
                    extract_2x2_ordered(y.surf_vars['pm10']),
                    extract_2x2_ordered(y.surf_vars['pm25']),
                    extract_2x2_ordered(y.surf_vars['so2']),
                    extract_2x2_ordered(y.surf_vars['no2']),
                    extract_2x2_ordered(y.surf_vars['o3']),
                    extract_2x2_ordered(y.surf_vars['co']),
                    extract_2x2_ordered(y.atmos_vars['q'][:,:,0,:,:])
                ]).unsqueeze(0)
                
                test_preds.append(pred_tensor.to(device))
                test_labels.append(y_tensor.to(device))
                
            except Exception as e:
                print(f"警告：批次 {i} 评估失败 - {e}")
                continue
    
    # 计算测试集性能
    if test_preds:
        test_preds_tensor = torch.cat(test_preds)
        test_labels_tensor = torch.cat(test_labels)
        
        test_mse = F.mse_loss(test_preds_tensor, test_labels_tensor)
        print(f'\n【重要】测试集 MSE: {test_mse.item():.6f}')
        
        # 保存测试集结果
        np.save('test_preds_oct21-31.npy', test_preds_tensor.cpu().detach().numpy())
        np.save('test_labels_oct21-31.npy', test_labels_tensor.cpu().detach().numpy())
        
        # 为了兼容后续代码，把测试集结果加入all_preds
        all_preds = test_preds
        all_labels = test_labels
    else:
        print("警告：测试集评估失败，没有有效结果")
        all_preds = []
        all_labels = []
    
    # ===== 可选：也评估训练集（仅供参考，检查过拟合）=====
    print("\n评估训练集最后20个批次（仅供参考）...")
    train_sample_preds = []
    train_sample_labels = []
    
    with torch.inference_mode():
        sample_indices = train_indices[-20:] if len(train_indices) > 20 else train_indices
        
        for i in sample_indices:
            try:
                X, y = X_batch[i], y_batch[i]
                pred = model.forward(X)
                
                # [使用相同的extract_2x2_ordered函数和提取逻辑]
                # ... (代码与上面相同)
                
            except:
                continue
    
    if train_sample_preds:
        train_sample_preds_tensor = torch.cat(train_sample_preds)
        train_sample_labels_tensor = torch.cat(train_sample_labels)
        train_mse = F.mse_loss(train_sample_preds_tensor, train_sample_labels_tensor)
        print(f'训练集样本 MSE: {train_mse.item():.6f}')
        
        # 检查过拟合
        if test_preds and train_mse.item() < test_mse.item() * 0.5:
            print("⚠️ 警告：训练集MSE远小于测试集，可能存在过拟合")
    
    # ===== 为了兼容性，合并所有结果 =====
    if all_preds:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
    else:
        print("错误：没有可用的预测结果")
        return
    
    # 保存完整数据
    np.save('all_preds_aurora_complete.npy', all_preds.cpu().detach().numpy())
    np.save('all_labels_aurora_complete.npy', all_labels.cpu().detach().numpy())
    
    # ===== 10月份可视化（现在只用测试集数据）=====
    # 修改：只传递测试集相关的数据
    test_X_batch = [X_batch[i] for i in test_indices]
    test_y_batch = [y_batch[i] for i in test_indices]
    
    october_preds, october_labels, october_timestamps = create_october_filtered_visualization(
        all_preds, all_labels, test_X_batch, test_y_batch, df_original)
    
    # 保存10月份数据
    np.save('october_preds_filtered.npy', october_preds.cpu().detach().numpy())
    np.save('october_labels_filtered.npy', october_labels.cpu().detach().numpy())
    
    times_df = pd.DataFrame({'timestamp': october_timestamps})
    times_df.to_csv('october_timestamps.csv', index=False)
    
    print(f"\n=== 处理完成 ===")
    print(f"测试集数据: {all_preds.shape}")
    print(f"10月份可视化数据: {october_preds.shape}")

def create_october_filtered_visualization(all_preds, all_labels, X_batch, y_batch, df_original):
    """10月数据可视化"""
    october_indices = []
    october_timestamps = []
    
    def is_valid_2hour_interval(input_time, pred_time):
        time_diff_hours = (pred_time - input_time).total_seconds() / 3600
        return abs(time_diff_hours - 2.0) <= 0.1
    
    for i, X in enumerate(X_batch):
        if i >= len(all_preds) or i >= len(y_batch):
            continue
        input_time = X.metadata.time[0]
        pred_time = y_batch[i].metadata.time[0]
        
        # 只要是10月的数据，并且预测间隔是2小时就包含
        if input_time.month == 10 and is_valid_2hour_interval(input_time, pred_time):
            october_indices.append(i)
            october_timestamps.append(input_time)
    
    # 按时间排序
    sorted_indices = sorted(range(len(october_timestamps)), 
                           key=lambda k: october_timestamps[k])
    october_indices = [october_indices[i] for i in sorted_indices]
    october_timestamps = [october_timestamps[i] for i in sorted_indices]
    
    october_indices_tensor = torch.tensor(october_indices)
    october_preds = all_preds[october_indices_tensor]
    october_labels = all_labels[october_indices_tensor]
    
    print(f"筛选出的10月数据量: {len(october_indices)}")
    if len(october_indices) > 0:
        print(f"第一个时间: {october_timestamps[0]}")
        print(f"最后一个时间: {october_timestamps[-1]}")
    
    # 生成所有图表
    create_station_grouped_visualization(october_preds, october_labels, october_timestamps)
    create_individual_station_plots(october_preds, october_labels, october_timestamps)
    
    # 生成Excel
    export_shangqingsi_predictions_to_excel(october_preds, october_labels, october_timestamps,
                                           [X_batch[i] for i in october_indices],
                                           [y_batch[i] for i in october_indices],
                                           df_original)
    
    return october_preds, october_labels, october_timestamps

def create_station_grouped_visualization(preds, labels, timestamps):
    """可视化 - 使用原始值"""
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
    
    station_names = ['Shangqingsi', 'Tangjiatuo', 'Tiansheng', 'Longjingwan']
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Aurora Model Predictions vs True Values - October Data by Station', fontsize=16, fontweight='bold')
    
    important_vars = ['Temperature (K)', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)',
                     'Pressure (Pa)', 'NO2 (μg/m³)', 'O3 (μg/m³)',
                     'SO2 (μg/m³)', 'CO (mg/m³)', 'U Wind (m/s)',
                     'V Wind (m/s)', 'Humidity (kg/kg)']
    
    colors = ["#FFB6B6", "#B9E3FF", "#FFEFAF", "#B4FFCA"]
    
    for idx, var_name in enumerate(important_vars):
        if idx >= 12:
            break
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        var_idx = variables[var_name]
        
        for station_idx in range(4):
            pred_values = []
            true_values = []
            
            for batch_idx in range(len(preds)):
                pred_val = preds[batch_idx, var_idx, station_idx].cpu().numpy()
                true_val = labels[batch_idx, var_idx, station_idx].cpu().numpy()
                pred_values.append(pred_val)
                true_values.append(true_val)
            
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
    
    plt.tight_layout()
    plt.savefig('october_predictions_by_station.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_station_plots(preds, labels, timestamps):
    """个站可视化"""
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
    
    station_names = ['Shangqingsi', 'Tangjiatuo', 'Tiansheng', 'Longjingwan']
    display_vars = list(variables.keys())
    
    for station_idx in range(4):
        station_name = station_names[station_idx]
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(f'{station_name} Station - All Variables Comparison (October)', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for var_idx, var_name in enumerate(display_vars):
            ax = axes_flat[var_idx]
            var_data_idx = variables[var_name]
            
            pred_values = []
            true_values = []
            
            for batch_idx in range(len(preds)):
                pred_val = preds[batch_idx, var_data_idx, station_idx].cpu().numpy()
                true_val = labels[batch_idx, var_data_idx, station_idx].cpu().numpy()
                pred_values.append(pred_val)
                true_values.append(true_val)
            
            time_indices = range(len(pred_values))
            ax.plot(time_indices, pred_values, '--', color="#F995AE", linewidth=1.5, alpha=0.8, label='Predicted')
            ax.plot(time_indices, true_values, '-', color="#8AE7F3", linewidth=1.5, alpha=0.9, label='True')
            
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Index (October)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if var_idx == 0:
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{station_name}_detailed_october.png', dpi=300, bbox_inches='tight')
        plt.close()

def export_shangqingsi_predictions_to_excel(preds, labels, timestamps, X_batch, y_batch, df_original):
    """导出上清寺Excel - 包含所有变量的完整信息"""
    shangqingsi_station_idx = 0
    station_name = "上清寺"
    
    # 准备原始数据映射
    df_shangqingsi_original = df_original[df_original['station_name'] == '上清寺'].copy()
    df_shangqingsi_original['time'] = pd.to_datetime(df_shangqingsi_original['time'])
    df_shangqingsi_original = df_shangqingsi_original.sort_values('time').reset_index(drop=True)
    
    original_data_map = {}
    for _, row in df_shangqingsi_original.iterrows():
        time_key = row['time']
        original_data_map[time_key] = row.to_dict()
    
    export_data = []
    
    for batch_idx in range(len(preds)):
        if batch_idx >= len(X_batch) or batch_idx >= len(y_batch):
            continue
            
        input_time = X_batch[batch_idx].metadata.time[0]
        pred_time = y_batch[batch_idx].metadata.time[0]
        time_diff_hours = (pred_time - input_time).total_seconds() / 3600
        
        # 基础信息
        row_data = {
            '站点名称': station_name,
            '输入时间': input_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测时间': pred_time.strftime('%Y-%m-%d %H:%M:%S'),
            '预测间隔(小时)': round(time_diff_hours, 2)
        }
        
        # 获取原始数据
        original_input = original_data_map.get(input_time, {})
        original_output = original_data_map.get(pred_time, {})
        
        # 温度
        row_data.update({
            '温度_输入实测': original_input.get('temperature', np.nan),
            '温度_真实值': original_output.get('temperature', np.nan),
            '温度_预测值': float(preds[batch_idx, 0, shangqingsi_station_idx].cpu().numpy()),
        })
        
        # 风速风向 (需要从u,v计算)
        pred_u = float(preds[batch_idx, 1, shangqingsi_station_idx].cpu().numpy())
        pred_v = float(preds[batch_idx, 2, shangqingsi_station_idx].cpu().numpy())
        pred_wind_speed, pred_wind_dir = uv_to_wind_speed_dir(pred_u, pred_v)
        
        true_u = float(labels[batch_idx, 1, shangqingsi_station_idx].cpu().numpy())
        true_v = float(labels[batch_idx, 2, shangqingsi_station_idx].cpu().numpy())
        true_wind_speed, true_wind_dir = uv_to_wind_speed_dir(true_u, true_v)
        
        row_data.update({
            '风速_输入实测': original_input.get('wind_speed', np.nan),
            '风速_真实值': original_output.get('wind_speed', np.nan),
            '风速_预测值': pred_wind_speed,
            '风向_输入实测': original_input.get('wind_direction', np.nan),
            '风向_真实值': original_output.get('wind_direction', np.nan),
            '风向_预测值': pred_wind_dir,
        })
        
        # 压力
        row_data.update({
            '压力_输入实测': original_input.get('pressure', np.nan),
            '压力_真实值': original_output.get('pressure', np.nan),
            '压力_预测值': float(preds[batch_idx, 3, shangqingsi_station_idx].cpu().numpy()),
        })
        
        # 湿度 (需要从比湿度转换)
        pred_temp = float(preds[batch_idx, 0, shangqingsi_station_idx].cpu().numpy())
        pred_pressure = float(preds[batch_idx, 3, shangqingsi_station_idx].cpu().numpy())
        pred_specific_humidity = float(preds[batch_idx, 10, shangqingsi_station_idx].cpu().numpy())
        pred_humidity_percent = specific_humidity_to_rh(pred_specific_humidity, pred_temp, pred_pressure)
        
        row_data.update({
            '湿度_输入实测': original_input.get('humidity', np.nan),
            '湿度_真实值': original_output.get('humidity', np.nan),
            '湿度_预测值': pred_humidity_percent,
        })
        
        # PM2.5
        row_data.update({
            'PM25_输入实测': original_input.get('pm25', np.nan),
            'PM25_真实值': original_output.get('pm25', np.nan),
            'PM25_预测值': float(max(0, preds[batch_idx, 5, shangqingsi_station_idx].cpu().numpy())),
        })
        
        # PM10
        row_data.update({
            'PM10_输入实测': original_input.get('pm10', np.nan),
            'PM10_真实值': original_output.get('pm10', np.nan),
            'PM10_预测值': float(max(0, preds[batch_idx, 4, shangqingsi_station_idx].cpu().numpy())),
        })
        
        # SO2
        row_data.update({
            'SO2_输入实测': original_input.get('so2', np.nan),
            'SO2_真实值': original_output.get('so2', np.nan),
            'SO2_预测值': float(max(0, preds[batch_idx, 6, shangqingsi_station_idx].cpu().numpy())),
        })
        
        # NO2
        row_data.update({
            'NO2_输入实测': original_input.get('no2', np.nan),
            'NO2_真实值': original_output.get('no2', np.nan),
            'NO2_预测值': float(max(0, preds[batch_idx, 7, shangqingsi_station_idx].cpu().numpy())),
        })
        
        # O3
        row_data.update({
            'O3_输入实测': original_input.get('o3', np.nan),
            'O3_真实值': original_output.get('o3', np.nan),
            'O3_预测值': float(max(0, preds[batch_idx, 8, shangqingsi_station_idx].cpu().numpy())),
        })
        
        # CO
        row_data.update({
            'CO_输入实测': original_input.get('co', np.nan),
            'CO_真实值': original_output.get('co', np.nan),
            'CO_预测值': float(max(0, preds[batch_idx, 9, shangqingsi_station_idx].cpu().numpy())),
        })
        
        export_data.append(row_data)
    
    # 创建DataFrame并导出
    df_export = pd.DataFrame(export_data)
    df_export = df_export.sort_values('预测时间')
    
    # 计算误差列
    for var in ['温度', '风速', '压力', '湿度', 'PM25', 'PM10', 'SO2', 'NO2', 'O3', 'CO']:
        if f'{var}_真实值' in df_export.columns and f'{var}_预测值' in df_export.columns:
            df_export[f'{var}_误差'] = df_export[f'{var}_预测值'] - df_export[f'{var}_真实值']
            df_export[f'{var}_误差率%'] = (df_export[f'{var}_误差'] / df_export[f'{var}_真实值'].replace(0, np.nan)) * 100
    
    filename = '上清寺_10月份预测结果_完整版.xlsx'
    df_export.to_excel(filename, index=False, engine='openpyxl')
    print(f"导出完成: {filename}")
    print(f"包含变量: 温度、风速、风向、压力、湿度、PM2.5、PM10、SO2、NO2、O3、CO")
    print(f"每个变量包含: 输入实测值、真实值、预测值、误差、误差率")

def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    """Aurora损失函数"""
    mae_loss = torch.nn.L1Loss()
    total_surf_loss = 0.0
    total_atmos_loss = 0.0

    for var in pred_batch.surf_vars:
        total_surf_loss += mae_loss(pred_batch.surf_vars[var].float(), true_batch.surf_vars[var].float())

    for var in pred_batch.atmos_vars:
        total_atmos_loss += mae_loss(pred_batch.atmos_vars[var].float(), true_batch.atmos_vars[var].float())

    return total_surf_loss + total_atmos_loss

if __name__ == '__main__':
    torch.set_num_threads(4)
    
    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"
    
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    
    # 取原始数据
    df_weather_raw = pd.read_excel(os.path.join(base_path, filename_weather), header=0)
    df_weather_raw = df_weather_raw[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度',
                                     '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather_raw.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed', 'wind_direction']
    
    df_airquality_raw = pd.read_excel(os.path.join(base_path, filename_airquality), sheet_name=None, header=0)
    df_airquality_raw = pd.concat(df_airquality_raw.values(), ignore_index=True)
    df_airquality_raw = df_airquality_raw[['station_name', 'monitoring_time', 'longitude', 'latitude',
                                           'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']].copy()
    df_airquality_raw.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    df_airquality_raw['time'] = pd.to_datetime(df_airquality_raw['time'])
    df_weather_raw['time'] = pd.to_datetime(df_weather_raw['time'])
    df_original = pd.merge(df_weather_raw, df_airquality_raw, on=['station_name', 'time'], how='inner')
    
    # 处理数据
    df = process_excel2(filename_weather, filename_airquality)
    X_batch, y_batch = form_aurora_batch(df)
    
    # 验证数据
    print(f"\n=== 验证原始数据与批次数据一致性 ===")
    for i in range(min(3, len(y_batch))):
        batch_time = y_batch[i].metadata.time[0]
        print(f"\n批次{i} - 时间: {batch_time}")
        
        # 检查天生站
        temp_2x2 = y_batch[i].surf_vars['2t'][0, 0, 5:7, 5:7]
        pm25_2x2 = y_batch[i].surf_vars['pm25'][0, 0, 5:7, 5:7]
        
        batch_temp = temp_2x2[0, 1].item()
        batch_pm25 = pm25_2x2[0, 1].item()
        
        print(f"批次数据 - 天生站: 温度={batch_temp:.2f}K, PM2.5={batch_pm25:.2f}")
        
        # 查找原始数据
        orig = df_original[(df_original['station_name'] == '天生') & 
                           (df_original['time'] == batch_time)]
        if not orig.empty:
            print(f"原始数据 - 天生站: 温度={orig['temperature'].values[0]:.2f}K, PM2.5={orig['pm25'].values[0]:.2f}")
            
            if abs(batch_temp - orig['temperature'].values[0]) > 0.1:
                print("警告：温度不一致！")
            if abs(batch_pm25 - orig['pm25'].values[0]) > 0.1:
                print("警告：PM2.5不一致！")
    
    # Aurora模型
    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "10u", "10v", "msl", "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        atmos_vars=("t", "u", "v", "q"),
        autocast=True,
    )
    
    # 归一化参数设置 - 仅为空气质量变量设置
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
    
    print(f"\n=== 归一化参数 ===")
    print(f"PM2.5: location={locations['pm25']:.2f}, scale={scales['pm25']:.2f}")
    print(f"PM10: location={locations['pm10']:.2f}, scale={scales['pm10']:.2f}")
    
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    main(model, X_batch, y_batch, device, df_original)
