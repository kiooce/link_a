import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from utils import process_excel2, form_aurora_batch
import numpy as np
import torch
import torch.nn.functional as F
from aurora import Aurora, rollout, Batch, Metadata
from aurora.normalisation import locations, scales

class BestModelManager:
    """管理最佳模型的保存和清理"""
    def __init__(self, max_models=5, mse_threshold=50.0):
        self.max_models = max_models  # 最多保存多少个模型
        self.mse_threshold = mse_threshold  # MSE阈值
        self.saved_models = []  # 保存的模型列表 [(epoch, mse, paths), ...]
        
    def should_save(self, mse):
        """判断是否应该保存当前模型"""
        if mse > self.mse_threshold:
            return False
        
        if len(self.saved_models) < self.max_models:
            return True
            
        # 检查是否比最差的模型好
        worst_mse = max(self.saved_models, key=lambda x: x[1])[1]
        return mse < worst_mse
    
    def add_model(self, epoch, mse, model_paths):
        """添加新模型，如果超过限制则删除最差的"""
        self.saved_models.append((epoch, mse, model_paths))
        self.saved_models.sort(key=lambda x: x[1])  # 按MSE排序
        
        # 如果超过最大数量，删除最差的模型
        while len(self.saved_models) > self.max_models:
            epoch_to_remove, mse_to_remove, paths_to_remove = self.saved_models.pop()
            self._remove_model_files(epoch_to_remove, paths_to_remove)
            print(f"🗑️  删除较差模型 Epoch {epoch_to_remove} (MSE: {mse_to_remove:.4f})")
    
    def _remove_model_files(self, epoch, paths):
        """删除模型相关的所有文件"""
        for file_path in paths['files_to_remove']:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"删除文件失败 {file_path}: {e}")
    
    def get_status(self):
        """获取当前保存状态"""
        if not self.saved_models:
            return "暂无保存的模型"
        
        status = f"已保存 {len(self.saved_models)} 个最佳模型:\n"
        for epoch, mse, _ in self.saved_models:
            status += f"  - Epoch {epoch}: MSE {mse:.4f}\n"
        return status

def save_predictions_to_excel(all_preds, all_labels, X_batch, y_batch, epoch, output_dir="prediction_results"):
    # [保持原有函数不变]
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
    
    # 返回保存的文件路径
    return {
        'detailed': detailed_filename,
        'summary': summary_filename,
        'comparison': comparison_filename,
        'summary_stats': summary_stats
    }

def create_summary_statistics(preds_np, labels_np, variable_names, epoch):
    # [保持原有函数不变]
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
    # [保持原有函数不变]
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
    # [保持原有函数不变]
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
    
    print(f"所有epoch对比结果已保存: {comparison_filename}")

def main(model, X_batch, y_batch, device):
    print("🚀 开始智能训练（只保存最佳模型）...")
    
    # 🎯 配置参数
    MSE_THRESHOLD = 50.0  # MSE阈值，可以调整
    MAX_MODELS = 3        # 最多保存3个最佳模型
    
    # 初始化目录和管理器
    weights_dir = Path("best_models")
    weights_dir.mkdir(exist_ok=True)
    npy_dir = Path("best_training_data")
    npy_dir.mkdir(exist_ok=True)
    
    model_manager = BestModelManager(max_models=MAX_MODELS, mse_threshold=MSE_THRESHOLD)

    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 200  # 现在可以训练更多epoch了
    
    # 基础指标追踪（始终保存）
    epoch_losses = []
    epoch_mses = []
    all_epoch_stats = []
    
    print(f"📊 设置: MSE阈值={MSE_THRESHOLD}, 最多保存{MAX_MODELS}个模型")

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

            optimizer.zero_grad()
            lat, lon = X.metadata.lat, X.metadata.lon
            pred = pred.to(device)
            y = y.to(device)

            loss = aurora_loss(pred, y, reg_weight_div=0.1, lat=lat, lon=lon)
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

        # 计算基础指标
        avg_epoch_loss = epoch_loss / len(X_batch)
        epoch_losses.append(avg_epoch_loss)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mse = F.mse_loss(all_preds, all_labels)
        epoch_mses.append(mse.item())
        
        # 🎯 智能保存逻辑
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, MSE: {mse.item():.4f}")
        
        if model_manager.should_save(mse.item()):
            print(f"✅ 发现优秀模型! MSE {mse.item():.4f} < 阈值 {MSE_THRESHOLD}")
            
            # 保存模型权重
            model_name = f"epoch_{epoch+1}_mse_{mse.item():.4f}.pth"
            model_path = weights_dir / model_name
            torch.save(model.state_dict(), model_path)
            
            # 保存numpy数据
            preds_path = npy_dir / f'epoch_{epoch+1}_preds.npy'
            labels_path = npy_dir / f'epoch_{epoch+1}_labels.npy'
            np.save(preds_path, all_preds.cpu().numpy())
            np.save(labels_path, all_labels.cpu().numpy())
            
            # 保存Excel结果
            excel_results = save_predictions_to_excel(
                all_preds, all_labels, X_batch, y_batch, epoch + 1
            )
            all_epoch_stats.append(excel_results['summary_stats'])
            
            # 记录保存的文件路径
            model_paths = {
                'files_to_remove': [
                    model_path, preds_path, labels_path,
                    excel_results['detailed'], 
                    excel_results['summary'], 
                    excel_results['comparison']
                ]
            }
            
            model_manager.add_model(epoch + 1, mse.item(), model_paths)
            
            print(f"📁 已保存模型文件:")
            print(f"  - 权重: {model_path}")
            print(f"  - 数据: {preds_path}, {labels_path}")
            print(f"  - Excel: {excel_results['detailed'].name}")
            
        else:
            if mse.item() > MSE_THRESHOLD:
                print(f"⏭️  跳过保存: MSE {mse.item():.4f} > 阈值 {MSE_THRESHOLD}")
            else:
                print(f"⏭️  跳过保存: 未进入最佳{MAX_MODELS}个模型")
        
        # 每10个epoch显示状态
        if (epoch + 1) % 10 == 0:
            print("\n" + "="*50)
            print(model_manager.get_status())
            print("="*50 + "\n")
        
        # 始终保存基础训练指标
        np.save(npy_dir / 'training_losses.npy', np.array(epoch_losses))
        np.save(npy_dir / 'training_mses.npy', np.array(epoch_mses))

    # 训练结束后的总结
    print("\n🎉 训练完成!")
    print(model_manager.get_status())
    
    # 保存最终的epoch对比（仅包含保存的模型）
    if all_epoch_stats:
        save_epoch_comparison(all_epoch_stats)
    
    # 运行最终评估（使用最佳模型）
    if model_manager.saved_models:
        best_epoch, best_mse, _ = model_manager.saved_models[0]  # 第一个是最佳的
        print(f"\n🏆 使用最佳模型进行最终评估: Epoch {best_epoch}, MSE {best_mse:.4f}")
        
        # 加载最佳模型
        best_model_path = weights_dir / f"epoch_{best_epoch}_mse_{best_mse:.4f}.pth"
        model.load_state_dict(torch.load(best_model_path))
        
    evaluation(model, X_batch, y_batch, device)

def evaluation(model, X_batch, y_batch, device):
    # [保持原有函数不变]
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
    mse = F.mse_loss(all_preds, all_labels)

    create_comprehensive_visualization(all_preds, all_labels)

    np.save('final_best_preds.npy', all_preds.cpu().numpy())
    np.save('final_best_labels.npy', all_labels.cpu().numpy())
    
    print(f'🏆 最终最佳模型 MSE: {mse.item():.4f}')

def create_comprehensive_visualization(all_preds, all_labels):
    # [保持原有函数不变]
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
    fig.suptitle('Predictions vs True Values for All Variables (All 144 Points)', fontsize=16, fontweight='bold')
    
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
            ax.set_xlabel('Point Index (All Samples × 144 Points)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            correlation = np.corrcoef(pred_values_all_points, true_values_all_points)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(len(variables), len(axes_flat)):
        axes_flat[idx].remove()
    
    plt.tight_layout()
    plt.savefig('best_model_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(16, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for idx, (var_name, var_idx) in enumerate(variables.items()):
        pred_values_all_points = []
        true_values_all_points = []
        
        for i in range(len(all_preds)):
            pred_vals = all_preds[i][0, var_idx].cpu().numpy().flatten()
            true_vals = all_labels[i][0, var_idx].cpu().numpy().flatten()
            pred_values_all_points.extend(pred_vals)
            true_values_all_points.extend(true_vals)
        
        point_indices = range(len(pred_values_all_points))
        plt.plot(point_indices, pred_values_all_points, '--', color=colors[idx % len(colors)], 
                linewidth=1.5, alpha=0.6, label=f'{var_name} (Pred)')
        plt.plot(point_indices, true_values_all_points, '-', color=colors[idx % len(colors)], 
                linewidth=1.5, alpha=0.8, label=f'{var_name} (True)')
    
    plt.xlabel('Point Index (All Samples × 144 Points)', fontsize=12)
    plt.ylabel('Normalized Values', fontsize=12)
    plt.title('Best Model: All Variables Predictions vs True Values', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('best_model_all_variables_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("最佳模型可视化已保存:")
    print("- best_model_predictions_vs_true.png")
    print("- best_model_all_variables_combined.png")

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

    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

    device = 'cuda'
    model = model.to(device)

    try:
        main(model, X_batch, y_batch, device)
    except:
        evaluation(model, X_batch, y_batch, device)
