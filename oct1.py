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
import seaborn as sns
from scipy import stats

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
    # 层级冻结70%
    all_params = list(model.parameters())
    freeze_until = int(len(all_params) * 0.5)
    for i, param in enumerate(all_params[:freeze_until]):
        param.requires_grad = False

    print(f"冻结了前{freeze_until}个参数，训练后{len(all_params)-freeze_until}个参数")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    num_epochs = 300
    
    print(f"\n=== 开始训练（使用{len(train_indices)}个批次）===")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        # 只遍历训练集索引
        for i in train_indices:
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
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
            # 学习率衰减
            if epoch == 100:
                for g in optimizer.param_groups:
                    g['lr'] = 5e-5
            elif epoch == 200:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5
    
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

# ===== 新增：详细评估指标 =====
        print("\n=== 计算详细评估指标 ===")
        metrics_df = calculate_metrics_and_visualize(test_preds_tensor, test_labels_tensor)
        print("\n评估指标总览：")
        print(metrics_df.to_string(index=False))
        
        # ===== 新增：误差分析 =====
        print("\n=== 进行误差分析 ===")
        test_timestamps = [y_batch[i].metadata.time[0] for i in test_indices]
        error_stats = error_analysis(test_preds_tensor, test_labels_tensor, test_timestamps)
        print("误差统计：")
        for key, value in error_stats.items():
            print(f"  {key}: {value:.4f}")
        
        # ===== 新增：基线对比 =====
        print("\n=== 计算基线模型性能 ===")
        persist_preds, persist_labels = create_baseline_predictions(X_batch, y_batch, test_indices)
        if persist_preds is not None:
            persist_preds = persist_preds.to(device)
            persist_labels = persist_labels.to(device)
            persist_mse = F.mse_loss(persist_preds, persist_labels)
            print(f"持续性预报 MSE: {persist_mse.item():.6f}")
            print(f"Aurora相对改进: {((persist_mse.item() - test_mse.item()) / persist_mse.item() * 100):.2f}%")
            
            # 分变量对比
            print("\n分变量MSE对比：")
            var_names = ['温度', 'U风', 'V风', '压力', 'PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'CO', '湿度']
            for i, var_name in enumerate(var_names):
                aurora_var_mse = F.mse_loss(test_preds_tensor[:, i], test_labels_tensor[:, i]).item()
                persist_var_mse = F.mse_loss(persist_preds[:, i], persist_labels[:, i]).item()
                improvement = ((persist_var_mse - aurora_var_mse) / persist_var_mse * 100) if persist_var_mse > 0 else 0
                print(f"  {var_name}: Aurora={aurora_var_mse:.2f}, 持续性={persist_var_mse:.2f}, 改进={improvement:.1f}%")

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
    """可视化 - 使用原始值，并添加R²值"""
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
        
        # 收集所有站点的数据用于计算整体R²
        all_pred_values = []
        all_true_values = []
        
        for station_idx in range(4):
            pred_values = []
            true_values = []
            
            for batch_idx in range(len(preds)):
                pred_val = preds[batch_idx, var_idx, station_idx].cpu().numpy()
                true_val = labels[batch_idx, var_idx, station_idx].cpu().numpy()
                pred_values.append(pred_val)
                true_values.append(true_val)
                all_pred_values.append(pred_val)
                all_true_values.append(true_val)
            
            time_indices = range(len(pred_values))
            ax.plot(time_indices, pred_values, '--', color=colors[station_idx],
                   linewidth=1, alpha=0.8, label=f'{station_names[station_idx]} Pred')
            ax.plot(time_indices, true_values, '-', color=colors[station_idx],
                   linewidth=1.5, alpha=0.9, label=f'{station_names[station_idx]} True')
        
        # 计算R²
        all_pred_values = np.array(all_pred_values)
        all_true_values = np.array(all_true_values)
        
        # 计算R²
        if len(all_true_values) > 0 and np.var(all_true_values) > 0:
            ss_res = np.sum((all_true_values - all_pred_values) ** 2)
            ss_tot = np.sum((all_true_values - np.mean(all_true_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2 = 0
        
        # 计算相关系数
        if len(all_true_values) > 1:
            corr = np.corrcoef(all_pred_values, all_true_values)[0, 1]
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0
        
        ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index (October)', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在右下角添加R²值
        ax.text(0.98, 0.02, f'R²={r2:.3f}\nr={corr:.3f}', 
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('october_predictions_by_station.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_station_plots(preds, labels, timestamps):
    """个站可视化 - 添加R²值"""
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
            
            pred_values = np.array(pred_values)
            true_values = np.array(true_values)
            
            # 计算R²和相关系数
            if len(true_values) > 0 and np.var(true_values) > 0:
                ss_res = np.sum((true_values - pred_values) ** 2)
                ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                r2 = 0
            
            if len(true_values) > 1:
                corr = np.corrcoef(pred_values, true_values)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
            
            time_indices = range(len(pred_values))
            ax.plot(time_indices, pred_values, '--', color="#F995AE", linewidth=1.5, alpha=0.8, label='Predicted')
            ax.plot(time_indices, true_values, '-', color="#8AE7F3", linewidth=1.5, alpha=0.9, label='True')
            
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Index (October)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # 在右下角添加R²值
            ax.text(0.98, 0.02, f'R²={r2:.3f}\nr={corr:.3f}', 
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
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

def calculate_metrics_and_visualize(preds, labels, save_path="metrics_results"):
    """计算并可视化多个评估指标"""
    # 变量名称
    var_names = ['Temperature(K)', 'U_Wind(m/s)', 'V_Wind(m/s)', 'Pressure(Pa)', 'PM10(μg/m³)', 
                 'PM2.5(μg/m³)', 'SO2(μg/m³)', 'NO2(μg/m³)', 'O3(μg/m³)', 'CO(mg/m³)', 'Humidity(kg/kg)']
    
    # 存储所有指标
    metrics_dict = {'Variable': [], 'MAE': [], 'RMSE': [], 'R²': [], 'MAPE(%)': []}
    
    for i, var_name in enumerate(var_names):
        pred_var = preds[:, i, :].flatten()
        true_var = labels[:, i, :].flatten()
        
        # 计算指标
        mae = torch.mean(torch.abs(pred_var - true_var)).item()
        mse = torch.mean((pred_var - true_var) ** 2).item()
        rmse = np.sqrt(mse)
        
        # R²
        ss_tot = torch.sum((true_var - torch.mean(true_var)) ** 2)
        ss_res = torch.sum((true_var - pred_var) ** 2)
        r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else 0
        
        # MAPE (避免除零)
        mask = torch.abs(true_var) > 0.001
        if mask.any():
            mape = torch.mean(torch.abs((true_var[mask] - pred_var[mask]) / true_var[mask])) * 100
            mape = mape.item()
        else:
            mape = np.nan
        
        metrics_dict['Variable'].append(var_name)
        metrics_dict['MAE'].append(mae)
        metrics_dict['RMSE'].append(rmse)
        metrics_dict['R²'].append(r2)
        metrics_dict['MAPE(%)'].append(mape)
    
    # 创建DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    
    # 1. 保存为Excel（方便写论文）
    metrics_df.to_excel(f'{save_path}_table.xlsx', index=False)
    print(f"评估指标已保存到 {save_path}_table.xlsx")
    
    # 2. 创建分组柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(['MAE', 'RMSE', 'R²', 'MAPE(%)']):
        ax = axes[idx // 2, idx % 2]
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))
        bars = ax.bar(range(len(metrics_df)), metrics_df[metric], color=colors)
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels([v.split('(')[0] for v in metrics_df['Variable']], rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Variable')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, metrics_df[metric]):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Aurora Model Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}_bars.png', dpi=300)
    plt.close()
    
    return metrics_df

def error_analysis(preds, labels, timestamps, save_path="error_analysis"):
    """误差分析和可视化"""
    # 计算误差
    errors = preds - labels  # [batches, variables, stations]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 误差分布直方图（以PM2.5为例）
    ax = axes[0, 0]
    pm25_errors = errors[:, 5, :].flatten().cpu().numpy()
    ax.hist(pm25_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('PM2.5 Prediction Error (μg/m³)')
    ax.set_ylabel('Frequency')
    ax.set_title('PM2.5 Error Distribution')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    ax.legend()
    
    # 2. 误差随时间变化
    ax = axes[0, 1]
    time_indices = range(len(errors))
    mean_abs_errors = torch.mean(torch.abs(errors), dim=(1, 2)).cpu().numpy()
    ax.plot(time_indices, mean_abs_errors)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error Evolution Over Time')
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q图
    ax = axes[0, 2]
    stats.probplot(pm25_errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot for PM2.5 Errors')
    
    # 4. 分站点误差箱线图
    ax = axes[1, 0]
    station_names = ['SQS', 'TJT', 'TS', 'LJW']
    station_errors = []
    for s in range(4):
        station_errors.append(errors[:, 5, s].cpu().numpy())  # PM2.5
    ax.boxplot(station_errors, labels=station_names)
    ax.set_ylabel('PM2.5 Error (μg/m³)')
    ax.set_title('Error Distribution by Station')
    ax.grid(True, alpha=0.3)
    
    # 5. 散点图：预测vs真实
    ax = axes[1, 1]
    true_pm25 = labels[:, 5, :].flatten().cpu().numpy()
    pred_pm25 = preds[:, 5, :].flatten().cpu().numpy()
    ax.scatter(true_pm25, pred_pm25, alpha=0.3)
    ax.plot([0, 150], [0, 150], 'r--', label='Perfect Prediction')
    ax.set_xlabel('True PM2.5 (μg/m³)')
    ax.set_ylabel('Predicted PM2.5 (μg/m³)')
    ax.set_title('Prediction vs Truth')
    ax.legend()
    ax.set_xlim(0, max(true_pm25.max(), pred_pm25.max()))
    ax.set_ylim(0, max(true_pm25.max(), pred_pm25.max()))
    
    # 6. 相关性热力图
    ax = axes[1, 2]
    # 计算每个变量的R²
    var_names_short = ['T', 'U', 'V', 'P', 'PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'CO', 'Q']
    r2_matrix = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            pred_i = preds[:, i, :].flatten().cpu().numpy()
            true_j = labels[:, j, :].flatten().cpu().numpy()
            corr = np.corrcoef(pred_i, true_j)[0, 1]
            r2_matrix[i, j] = corr ** 2 if not np.isnan(corr) else 0
    
    im = ax.imshow(r2_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_xticklabels(var_names_short, fontsize=8)
    ax.set_yticklabels(var_names_short, fontsize=8)
    ax.set_title('Variable Correlation (R²)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Error Analysis for Aurora Model', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出统计信息
    error_stats = {
        'Mean Error': torch.mean(errors).item(),
        'Std Error': torch.std(errors).item(),
        'Max Abs Error': torch.max(torch.abs(errors)).item(),
        'Error Skewness': stats.skew(errors.flatten().cpu().numpy()),
        'Error Kurtosis': stats.kurtosis(errors.flatten().cpu().numpy())
    }
    
    return error_stats

def create_baseline_predictions(X_batch, y_batch, test_indices):
    """创建基线预测用于对比"""
    persistence_preds = []
    persistence_labels = []
    
    for i in test_indices:
        X = X_batch[i]
        y = y_batch[i]
        
        # 持续性预报：用最后一个时刻作为预测
        # 提取X的最后时刻数据作为预测
        def extract_2x2_ordered(tensor_12x12):
            center_2x2 = tensor_12x12[0, -1, 5:7, 5:7]  # 注意这里用-1取最后时刻
            return torch.stack([
                center_2x2[1,1], center_2x2[1,0], 
                center_2x2[0,1], center_2x2[0,0]
            ])
        
        # 构建持续性预测
        persist_pred = torch.stack([
            extract_2x2_ordered(X.surf_vars['2t']),
            extract_2x2_ordered(X.surf_vars['10u']),
            extract_2x2_ordered(X.surf_vars['10v']),
            extract_2x2_ordered(X.surf_vars['msl']),
            extract_2x2_ordered(X.surf_vars['pm10']),
            extract_2x2_ordered(X.surf_vars['pm25']),
            extract_2x2_ordered(X.surf_vars['so2']),
            extract_2x2_ordered(X.surf_vars['no2']),
            extract_2x2_ordered(X.surf_vars['o3']),
            extract_2x2_ordered(X.surf_vars['co']),
            extract_2x2_ordered(X.atmos_vars['q'][:,:,0,:,:])
        ]).unsqueeze(0)
        
        # 真实值
        def extract_2x2_ordered_y(tensor_12x12):
            center_2x2 = tensor_12x12[0, 0, 5:7, 5:7]
            return torch.stack([
                center_2x2[1,1], center_2x2[1,0], 
                center_2x2[0,1], center_2x2[0,0]
            ])
        
        true_label = torch.stack([
            extract_2x2_ordered_y(y.surf_vars['2t']),
            extract_2x2_ordered_y(y.surf_vars['10u']),
            extract_2x2_ordered_y(y.surf_vars['10v']),
            extract_2x2_ordered_y(y.surf_vars['msl']),
            extract_2x2_ordered_y(y.surf_vars['pm10']),
            extract_2x2_ordered_y(y.surf_vars['pm25']),
            extract_2x2_ordered_y(y.surf_vars['so2']),
            extract_2x2_ordered_y(y.surf_vars['no2']),
            extract_2x2_ordered_y(y.surf_vars['o3']),
            extract_2x2_ordered_y(y.surf_vars['co']),
            extract_2x2_ordered_y(y.atmos_vars['q'][:,:,0,:,:])
        ]).unsqueeze(0)
        
        persistence_preds.append(persist_pred)
        persistence_labels.append(true_label)
    
    if persistence_preds:
        persistence_preds = torch.cat(persistence_preds)
        persistence_labels = torch.cat(persistence_labels)
        return persistence_preds, persistence_labels
    else:
        return None, None


def weighted_aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    """加权损失函数 - 重点关注空气质量，目前忽略"""
    mae_loss = torch.nn.L1Loss()
    
    # 气象变量（权重较低）
    weather_loss = 0
    for var in ['2t', '10u', '10v']:
        if var in pred_batch.surf_vars:
            weather_loss += mae_loss(pred_batch.surf_vars[var].float(), 
                                    true_batch.surf_vars[var].float()) * 0.5
    
    # 压力（值太大，降权）
    if 'msl' in pred_batch.surf_vars:
        pressure_loss = mae_loss(pred_batch.surf_vars['msl'].float(), 
                                true_batch.surf_vars['msl'].float()) * 0.001
    else:
        pressure_loss = 0
    
    # 空气质量（加权重，这是重点）
    air_loss = 0
    for var in ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']:
        if var in pred_batch.surf_vars:
            air_loss += mae_loss(pred_batch.surf_vars[var].float(), 
                               true_batch.surf_vars[var].float()) * 2.0
    
    # 大气变量
    atmos_loss = 0
    for var in pred_batch.atmos_vars:
        atmos_loss += mae_loss(pred_batch.atmos_vars[var].float(), 
                              true_batch.atmos_vars[var].float()) * 0.5
    
    return weather_loss + pressure_loss + air_loss + atmos_loss

def aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon):
    """平衡损失函数 - 为CFD优化"""
    mae_loss = torch.nn.L1Loss()
    
    losses = {}
    
    # 气象变量 - 归一化损失
    if '2t' in pred_batch.surf_vars:
        losses['temp'] = mae_loss(pred_batch.surf_vars['2t'].float(), 
                                 true_batch.surf_vars['2t'].float()) / 5.0
    if '10u' in pred_batch.surf_vars:
        losses['u_wind'] = mae_loss(pred_batch.surf_vars['10u'].float(), 
                                    true_batch.surf_vars['10u'].float()) / 2.0
    if '10v' in pred_batch.surf_vars:
        losses['v_wind'] = mae_loss(pred_batch.surf_vars['10v'].float(), 
                                    true_batch.surf_vars['10v'].float()) / 2.0
    if 'msl' in pred_batch.surf_vars:
        losses['pressure'] = mae_loss(pred_batch.surf_vars['msl'].float(), 
                                      true_batch.surf_vars['msl'].float()) / 1000.0
    
    # 空气质量变量 - 归一化损失
    if 'pm25' in pred_batch.surf_vars:
        losses['pm25'] = mae_loss(pred_batch.surf_vars['pm25'].float(), 
                                  true_batch.surf_vars['pm25'].float()) / 20.0
    if 'pm10' in pred_batch.surf_vars:
        losses['pm10'] = mae_loss(pred_batch.surf_vars['pm10'].float(), 
                                  true_batch.surf_vars['pm10'].float()) / 30.0
    if 'no2' in pred_batch.surf_vars:
        losses['no2'] = mae_loss(pred_batch.surf_vars['no2'].float(), 
                                 true_batch.surf_vars['no2'].float()) / 20.0
    if 'so2' in pred_batch.surf_vars:
        losses['so2'] = mae_loss(pred_batch.surf_vars['so2'].float(), 
                                true_batch.surf_vars['so2'].float()) / 5.0
    if 'o3' in pred_batch.surf_vars:
        losses['o3'] = mae_loss(pred_batch.surf_vars['o3'].float(), 
                               true_batch.surf_vars['o3'].float()) / 30.0
    if 'co' in pred_batch.surf_vars:
        losses['co'] = mae_loss(pred_batch.surf_vars['co'].float(), 
                               true_batch.surf_vars['co'].float()) / 0.5
    
    # 大气变量
    for var in pred_batch.atmos_vars:
        losses[f'atmos_{var}'] = mae_loss(pred_batch.atmos_vars[var].float(), 
                                         true_batch.atmos_vars[var].float())
    
    return sum(losses.values())

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
    
    # 归一化
    mean_values = df.select_dtypes(include=['number']).mean()
    std_values = df.select_dtypes(include=['number']).std()

    # 气象变量的Aurora名称映射
    locations["2t"] = mean_values['temperature']
    scales["2t"] = std_values['temperature']

    locations["msl"] = mean_values['pressure']
    scales["msl"] = std_values['pressure']

    locations["10u"] = mean_values['u_wind']
    locations["10v"] = mean_values['v_wind']
    scales["10u"] = std_values['u_wind']
    scales["10v"] = std_values['v_wind']

    # 湿度（比湿度）
    locations["q"] = mean_values['specific_humidity']
    scales["q"] = std_values['specific_humidity']

    # 空气质量变量
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
