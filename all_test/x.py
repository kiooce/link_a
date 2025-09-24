#main_aurora_train.py
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from utils import process_excel2, form_aurora_batch
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

class DataNormalizer:
    """MinMax标准化类"""
    def __init__(self):
        self.scalers = {}
        self.feature_names = ['temperature', 'pm25', 'pm10', 'so2', 'no2', 'o3', 'humidity']
    
    def fit(self, df):
        """计算标准化参数"""
        for feature in self.feature_names:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                self.scalers[feature] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
        
        print("标准化参数计算完成")
        for feature, params in self.scalers.items():
            print(f"  {feature}: [{params['min']:.2f}, {params['max']:.2f}]")
    
    def transform(self, df):
        """应用标准化"""
        df_normalized = df.copy()
        for feature in self.feature_names:
            if feature in df_normalized.columns and feature in self.scalers:
                scaler = self.scalers[feature]
                if scaler['range'] > 0:
                    df_normalized[feature] = (df_normalized[feature] - scaler['min']) / scaler['range']
                else:
                    df_normalized[feature] = 0.0
        return df_normalized
    
    def inverse_transform_tensor(self, tensor, feature_idx):
        """反标准化tensor (用于评估)"""
        feature_name = self.feature_names[feature_idx]
        if feature_name in self.scalers:
            scaler = self.scalers[feature_name]
            return tensor * scaler['range'] + scaler['min']
        return tensor

class BestModelManager:
    """挑最好的几个case保存"""
    def __init__(self, max_models=3, mse_threshold=10.0):
        self.max_models = max_models
        self.mse_threshold = mse_threshold
        self.saved_models = []
        
    def should_save(self, mse):
        if mse > self.mse_threshold:
            return False
        if len(self.saved_models) < self.max_models:
            return True
        worst_mse = max(self.saved_models, key=lambda x: x[1])[1]
        return mse < worst_mse
    
    def add_model(self, epoch, mse, model_paths):
        self.saved_models.append((epoch, mse, model_paths))
        self.saved_models.sort(key=lambda x: x[1])
        
        while len(self.saved_models) > self.max_models:
            epoch_to_remove, mse_to_remove, paths_to_remove = self.saved_models.pop()
            self._remove_model_files(epoch_to_remove, paths_to_remove)
    
    def _remove_model_files(self, epoch, paths):
        for file_path in paths['files_to_remove']:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                pass
    
    def get_status(self):
        if not self.saved_models:
            return "无保存模型"
        
        status = f"已保存{len(self.saved_models)}个最佳模型:\n"
        for epoch, mse, _ in self.saved_models:
            status += f"  Epoch {epoch}: MSE {mse:.4f}\n"
        return status

def get_station_indices(verbose=False):
    station_positions = [
        (5, 6),  # 唐家沱 66
        (6, 4),  # 龙井湾 76 
        (4, 5),  # 天生 53
        (6, 6),  # 上清寺 78
        (7, 5),  # 鱼新街 89
    ]
    
    station_indices = [row * 12 + col for row, col in station_positions]
    
    if verbose:
        station_names = ['唐家沱', '龙井湾', '天生', '上清寺', '鱼新街']
        print("站点映射:")
        for i, (name, (row, col), idx) in enumerate(zip(station_names, station_positions, station_indices)):
            print(f"  {name}: 网格({row},{col}) -> 索引{idx}")
    
    return station_indices

def save_predictions_to_excel(all_preds, all_labels, X_batch, y_batch, epoch, output_dir="prediction_results"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    variable_names = ['Temperature', 'PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'Humidity']
    
    preds_np = all_preds.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    
    detailed_results = []
    
    for sample_idx in range(len(preds_np)):
        if hasattr(X_batch[sample_idx].metadata, 'time') and X_batch[sample_idx].metadata.time:
            input_timestamp = X_batch[sample_idx].metadata.time[0]
        else:
            input_timestamp = f"Input_Sample_{sample_idx}"
            
        if hasattr(y_batch[sample_idx].metadata, 'time') and y_batch[sample_idx].metadata.time:
            prediction_timestamp = y_batch[sample_idx].metadata.time[0]
        else:
            prediction_timestamp = f"Prediction_Sample_{sample_idx}"
        
        input_time_str = str(input_timestamp)
        prediction_time_str = str(prediction_timestamp)
        
        try:
            if hasattr(input_timestamp, 'strftime'):
                input_time_str = input_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            if hasattr(prediction_timestamp, 'strftime'):
                prediction_time_str = prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        for spatial_idx in range(144):
            row_data = {
                'Epoch': epoch,
                'Sample_Index': sample_idx,
                'Input_Time': input_time_str,
                'Prediction_Time': prediction_time_str,
                'Spatial_Point': spatial_idx,
            }
            
            for var_idx, var_name in enumerate(variable_names):
                pred_value = preds_np[sample_idx, var_idx, spatial_idx]
                true_value = labels_np[sample_idx, var_idx, spatial_idx]
                
                row_data[f'{var_name}_Predicted'] = pred_value
                row_data[f'{var_name}_True'] = true_value
                row_data[f'{var_name}_Error'] = abs(pred_value - true_value)
                row_data[f'{var_name}_Relative_Error'] = abs(pred_value - true_value) / (abs(true_value) + 1e-8) * 100
            
            detailed_results.append(row_data)
    
    df_detailed = pd.DataFrame(detailed_results)
    
    detailed_filename = output_path / f"epoch_{epoch}_detailed_predictions.xlsx"
    df_detailed.to_excel(detailed_filename, index=False)
    
    summary_stats = create_summary_statistics(preds_np, labels_np, variable_names, epoch)
    summary_filename = output_path / f"epoch_{epoch}_summary_stats.xlsx"
    summary_stats.to_excel(summary_filename, index=False)
    
    comparison_df = create_variable_comparison(preds_np, labels_np, variable_names, epoch)
    comparison_filename = output_path / f"epoch_{epoch}_variable_comparison.xlsx"
    comparison_df.to_excel(comparison_filename, index=False)
    
    return {
        'detailed': detailed_filename,
        'summary': summary_filename,
        'comparison': comparison_filename,
        'summary_stats': summary_stats
    }

def create_summary_statistics(preds_np, labels_np, variable_names, epoch):
    summary_data = []
    
    for var_idx, var_name in enumerate(variable_names):
        pred_values = preds_np[:, var_idx, :].flatten()
        true_values = labels_np[:, var_idx, :].flatten()
        
        mse = np.mean((pred_values - true_values) ** 2)
        mae = np.mean(np.abs(pred_values - true_values))
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0
        
        relative_error = np.mean(np.abs(pred_values - true_values) / (np.abs(true_values) + 1e-8)) * 100
        
        summary_data.append({
            'Epoch': epoch,
            'Variable': var_name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Mean_Relative_Error_%': relative_error,
            'Pred_Mean': np.mean(pred_values),
            'True_Mean': np.mean(true_values),
            'Pred_Std': np.std(pred_values),
            'True_Std': np.std(true_values),
            'Min_Error': np.min(np.abs(pred_values - true_values)),
            'Max_Error': np.max(np.abs(pred_values - true_values))
        })
    
    return pd.DataFrame(summary_data)

def create_variable_comparison(preds_np, labels_np, variable_names, epoch):
    comparison_data = {'Epoch': epoch, 'Sample_Index': [], 'Spatial_Point': []}
    
    for var_name in variable_names:
        comparison_data[f'{var_name}_Pred'] = []
        comparison_data[f'{var_name}_True'] = []
    
    for sample_idx in range(len(preds_np)):
        for spatial_idx in range(144):
            comparison_data['Sample_Index'].append(sample_idx)
            comparison_data['Spatial_Point'].append(spatial_idx)
            
            for var_idx, var_name in enumerate(variable_names):
                comparison_data[f'{var_name}_Pred'].append(preds_np[sample_idx, var_idx, spatial_idx])
                comparison_data[f'{var_name}_True'].append(labels_np[sample_idx, var_idx, spatial_idx])
    
    return pd.DataFrame(comparison_data)

def save_epoch_comparison(all_epoch_stats, output_dir="prediction_results"):
    output_path = Path(output_dir)
    comparison_filename = output_path / "all_epochs_comparison.xlsx"
    
    combined_stats = pd.concat(all_epoch_stats, ignore_index=True)
    
    pivot_table = combined_stats.pivot_table(
        index='Variable', 
        columns='Epoch', 
        values=['MSE', 'MAE', 'Correlation'], 
        aggfunc='first'
    )
    
    with pd.ExcelWriter(comparison_filename) as writer:
        combined_stats.to_excel(writer, sheet_name='All_Stats', index=False)
        pivot_table.to_excel(writer, sheet_name='Epoch_Comparison')

def weighted_aurora_loss(pred_batch, true_batch, reg_weight_div, lat, lon, station_weight=15.0):
    """带位置权重的Aurora损失函数"""
    mae_loss = torch.nn.L1Loss(reduction='none')
    
    station_indices = get_station_indices(verbose=False)
    
    spatial_weights = torch.ones(144, device=pred_batch.surf_vars['2t'].device)
    for idx in station_indices:
        spatial_weights[idx] = station_weight
    
    total_surf_loss = 0.0
    total_atmos_loss = 0.0
    
    for var in pred_batch.surf_vars:
        pointwise_loss = mae_loss(
            pred_batch.surf_vars[var].float().reshape(-1, 144), 
            true_batch.surf_vars[var].float().reshape(-1, 144)
        )
        weighted_loss = pointwise_loss * spatial_weights.unsqueeze(0)
        total_surf_loss += weighted_loss.mean()
    
    for var in pred_batch.atmos_vars:
        pred_reshaped = pred_batch.atmos_vars[var].float().reshape(-1, 144)
        true_reshaped = true_batch.atmos_vars[var].float().reshape(-1, 144)
        
        pointwise_loss = mae_loss(pred_reshaped, true_reshaped)
        weighted_loss = pointwise_loss * spatial_weights.unsqueeze(0)
        total_atmos_loss += weighted_loss.mean()
    
    div_free_loss = 0
    total_loss = total_surf_loss + total_atmos_loss + reg_weight_div * div_free_loss
    
    return total_loss

def main(model, X_batch, y_batch, device, normalizer):
    print("开始标准化加权训练...")
    
    MSE_THRESHOLD = 8.0  # 更严格的标准
    MAX_MODELS = 3
    STATION_WEIGHT = 15.0
    
    weights_dir = Path("best_models")
    weights_dir.mkdir(exist_ok=True)
    npy_dir = Path("best_training_data")
    npy_dir.mkdir(exist_ok=True)
    
    model_manager = BestModelManager(max_models=MAX_MODELS, mse_threshold=MSE_THRESHOLD)
    
    print(f"配置: 站点权重={STATION_WEIGHT}x, MSE目标<{MSE_THRESHOLD}")
    
    station_indices = get_station_indices(verbose=True)
    
    station_weight_total = len(station_indices) * STATION_WEIGHT
    other_weight_total = (144 - len(station_indices)) * 1
    station_proportion = station_weight_total / (station_weight_total + other_weight_total)
    print(f"权重分析: 5个站点占比 {station_proportion:.1%}")

    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 200
    
    epoch_losses = []
    epoch_mses = []
    all_epoch_stats = []

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
                    raise Exception("NaN detected in pred.surf")

            optimizer.zero_grad()
            lat, lon = X.metadata.lat, X.metadata.lon
            pred = pred.to(device)
            y = y.to(device)

            loss = weighted_aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon, station_weight=STATION_WEIGHT)
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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

        avg_epoch_loss = epoch_loss / len(X_batch)
        epoch_losses.append(avg_epoch_loss)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        overall_mse = F.mse_loss(all_preds, all_labels)
        
        station_preds = all_preds[:, :, station_indices]
        station_labels = all_labels[:, :, station_indices]
        station_mse = F.mse_loss(station_preds, station_labels)
        
        epoch_mses.append(overall_mse.item())
        
        # 简化输出：每10个epoch显示详细信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, 整体MSE={overall_mse.item():.4f}, 站点MSE={station_mse.item():.4f}")
        else:
            print(f"Epoch {epoch+1}: MSE={overall_mse.item():.4f}")
        
        if model_manager.should_save(overall_mse.item()):
            print(f"保存优秀模型: 整体MSE={overall_mse.item():.4f}, 站点MSE={station_mse.item():.4f}")
            
            model_name = f"epoch_{epoch+1}_mse_{overall_mse.item():.4f}.pth"
            model_path = weights_dir / model_name
            torch.save(model.state_dict(), model_path)
            
            preds_path = npy_dir / f'epoch_{epoch+1}_preds.npy'
            labels_path = npy_dir / f'epoch_{epoch+1}_labels.npy'
            np.save(preds_path, all_preds.cpu().numpy())
            np.save(labels_path, all_labels.cpu().numpy())
            
            excel_results = save_predictions_to_excel(
                all_preds, all_labels, X_batch, y_batch, epoch + 1
            )
            all_epoch_stats.append(excel_results['summary_stats'])
            
            model_paths = {
                'files_to_remove': [
                    model_path, preds_path, labels_path,
                    excel_results['detailed'], 
                    excel_results['summary'], 
                    excel_results['comparison']
                ]
            }
            
            model_manager.add_model(epoch + 1, overall_mse.item(), model_paths)
        
        if (epoch + 1) % 20 == 0:
            print(model_manager.get_status())
        
        np.save(npy_dir / 'training_losses.npy', np.array(epoch_losses))
        np.save(npy_dir / 'training_mses.npy', np.array(epoch_mses))

    print("训练结束")
    print(model_manager.get_status())
    
    if all_epoch_stats:
        save_epoch_comparison(all_epoch_stats)
    
    if model_manager.saved_models:
        best_epoch, best_mse, _ = model_manager.saved_models[0]
        print(f"最佳模型: Epoch {best_epoch}, MSE {best_mse:.4f}")
        
        best_model_path = weights_dir / f"epoch_{best_epoch}_mse_{best_mse:.4f}.pth"
        model.load_state_dict(torch.load(best_model_path))
        
    evaluation(model, X_batch, y_batch, device)

def evaluation(model, X_batch, y_batch, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for i in range(len(X_batch)):
            X, y = X_batch[i], y_batch[i]
            pred = model.forward(X)

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

            all_preds.append(pred_tensor.to(device))
            all_labels.append(y_tensor.to(device))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    overall_mse = F.mse_loss(all_preds, all_labels)
    
    station_indices = get_station_indices(verbose=False)
    station_preds = all_preds[:, :, station_indices]
    station_labels = all_labels[:, :, station_indices]
    station_mse = F.mse_loss(station_preds, station_labels)

    create_comprehensive_visualization(all_preds, all_labels)

    np.save('final_best_preds.npy', all_preds.cpu().numpy())
    np.save('final_best_labels.npy', all_labels.cpu().numpy())
    
    print(f'最终结果: 整体MSE={overall_mse.item():.4f}, 站点MSE={station_mse.item():.4f}')

def create_comprehensive_visualization(all_preds, all_labels):
    variables = {
        '2t (Temperature)': 0,
        'PM10': 1, 
        'PM2.5': 2,
        'SO2': 3,
        'NO2': 4,
        'O3': 5,
        'q (Humidity)': 6
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Normalized Weighted Training: Predictions vs True Values', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    pred_color = '#FF6B6B'
    true_color = '#4ECDC4'
    
    for idx, (var_name, var_idx) in enumerate(variables.items()):
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            
            pred_values_all_points = []
            true_values_all_points = []
            
            for i in range(len(all_preds)):
                pred_vals = all_preds[i][0, var_idx].cpu().numpy().flatten()
                true_vals = all_labels[i][0, var_idx].cpu().numpy().flatten()
                
                pred_values_all_points.extend(pred_vals)
                true_values_all_points.extend(true_vals)
            
            point_indices = range(len(pred_values_all_points))
            ax.plot(point_indices, pred_values_all_points, color=pred_color, linewidth=1, 
                   alpha=0.7, label='Predicted')
            ax.plot(point_indices, true_values_all_points, color=true_color, linewidth=1, 
                   alpha=0.7, label='True')
            
            ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Point Index', fontsize=10)
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            correlation = np.corrcoef(pred_values_all_points, true_values_all_points)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(len(variables), len(axes_flat)):
        axes_flat[idx].remove()
    
    plt.tight_layout()
    plt.savefig('normalized_weighted_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("可视化已保存: normalized_weighted_predictions.png")

if __name__ == '__main__':
    torch.set_num_threads(4)

    filename_weather = "5站点气象数据20231001-1231.xlsx"
    filename_airquality = "5站点202309-10月空气质量数据.xlsx"

    # 处理原始数据
    df = process_excel2(filename_weather, filename_airquality)
    
    # 创建和应用标准化器
    normalizer = DataNormalizer()
    normalizer.fit(df)
    df_normalized = normalizer.transform(df)
    
    print("数据标准化完成")
    print(f"标准化前数据范围示例: temp {df['temperature'].min():.2f}-{df['temperature'].max():.2f}")
    print(f"标准化后数据范围示例: temp {df_normalized['temperature'].min():.2f}-{df_normalized['temperature'].max():.2f}")
    
    # 检查标准化后的数据
    print("标准化后数据检查:")
    print(f"数据形状: {df_normalized.shape}")
    print(f"是否有NaN: {df_normalized.isnull().sum().sum()}")
    print(f"是否有无穷值: {np.isinf(df_normalized.select_dtypes(include=[np.number])).sum().sum()}")
    
    # 使用标准化数据生成batch
    try:
        X_batch, y_batch = form_aurora_batch(df_normalized)
        print(f"成功创建了 {len(X_batch)} 个batches")
    except Exception as e:
        print(f"创建batch时出错: {e}")
        print("回退到原始数据...")
        X_batch, y_batch = form_aurora_batch(df)
        normalizer = None  # 禁用标准化器

    model = Aurora(
        use_lora=False,
        surf_vars=("2t", "msl",  "pm10", "pm25", "so2", "no2", "o3", "co"),
        static_vars=("lsm", "slt"),
        atmos_vars=("t", "q"),
        autocast=True,
    )

    # 使用标准化后的统计信息（如果标准化成功）
    if normalizer is not None:
        mean_values = df_normalized.select_dtypes(include=['number']).mean()
        std_values = df_normalized.select_dtypes(include=['number']).std()
        print("使用标准化数据的统计信息")
    else:
        mean_values = df.select_dtypes(include=['number']).mean()
        std_values = df.select_dtypes(include=['number']).std()
        print("使用原始数据的统计信息")

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

    device = 'cuda'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device, normalizer)
    except Exception as e:
        print(f"训练出错: {e}")
        evaluation(model, X_batch, y_batch, device)
