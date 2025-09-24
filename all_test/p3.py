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

def save_predictions_to_excel(all_preds, all_labels, X_batch, y_batch, epoch, output_dir="prediction_results"):
    """
    将预测结果保存到Excel文件中
    
    Args:
        all_preds: 预测结果 tensor [样本数, 7, 144]
        all_labels: 真实标签 tensor [样本数, 7, 144] 
        X_batch: 输入batch，包含时间信息
        y_batch: 目标batch，包含预测时间信息
        epoch: 当前epoch数
        output_dir: 输出目录
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 变量名称
    variable_names = ['Temperature', 'PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'Humidity']
    
    # 转换为numpy
    preds_np = all_preds.cpu().numpy()  # [样本数, 7, 144]
    labels_np = all_labels.cpu().numpy()  # [样本数, 7, 144]
    
    # 创建详细结果DataFrame
    detailed_results = []
    
    for sample_idx in range(len(preds_np)):
        # 获取输入时间信息
        if hasattr(X_batch[sample_idx].metadata, 'time') and X_batch[sample_idx].metadata.time:
            input_timestamp = X_batch[sample_idx].metadata.time[0]
        else:
            input_timestamp = f"Input_Sample_{sample_idx}"
            
        # 获取预测目标时间信息
        if hasattr(y_batch[sample_idx].metadata, 'time') and y_batch[sample_idx].metadata.time:
            prediction_timestamp = y_batch[sample_idx].metadata.time[0]
        else:
            prediction_timestamp = f"Prediction_Sample_{sample_idx}"
        
        # 转换时间格式为更友好的显示
        input_time_str = str(input_timestamp)
        prediction_time_str = str(prediction_timestamp)
        
        # 如果是datetime对象，格式化为年-月-日 时:分:秒
        try:
            if hasattr(input_timestamp, 'strftime'):
                input_time_str = input_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            if hasattr(prediction_timestamp, 'strftime'):
                prediction_time_str = prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        # 遍历每个空间点
        for spatial_idx in range(144):
            row_data = {
                'Epoch': epoch,
                'Sample_Index': sample_idx,
                'Input_Time': input_time_str,
                'Prediction_Time': prediction_time_str,
                'Spatial_Point': spatial_idx,
            }
            
            # 添加每个变量的预测值和真实值
            for var_idx, var_name in enumerate(variable_names):
                pred_value = preds_np[sample_idx, var_idx, spatial_idx]
                true_value = labels_np[sample_idx, var_idx, spatial_idx]
                
                row_data[f'{var_name}_Predicted'] = pred_value
                row_data[f'{var_name}_True'] = true_value
                row_data[f'{var_name}_Error'] = abs(pred_value - true_value)
                row_data[f'{var_name}_Relative_Error'] = abs(pred_value - true_value) / (abs(true_value) + 1e-8) * 100
            
            detailed_results.append(row_data)
    
    # 创建DataFrame
    df_detailed = pd.DataFrame(detailed_results)
    
    # 保存详细结果
    detailed_filename = output_path / f"epoch_{epoch}_detailed_predictions.xlsx"
    df_detailed.to_excel(detailed_filename, index=False)
    
    # 创建汇总统计
    summary_stats = create_summary_statistics(preds_np, labels_np, variable_names, epoch)
    summary_filename = output_path / f"epoch_{epoch}_summary_stats.xlsx"
    summary_stats.to_excel(summary_filename, index=False)
    
    # 创建变量对比表
    comparison_df = create_variable_comparison(preds_np, labels_np, variable_names, epoch)
    comparison_filename = output_path / f"epoch_{epoch}_variable_comparison.xlsx"
    comparison_df.to_excel(comparison_filename, index=False)
    
    print(f"Epoch {epoch} 预测结果已保存:")
    print(f"  - 详细结果: {detailed_filename}")
    print(f"  - 汇总统计: {summary_filename}")
    print(f"  - 变量对比: {comparison_filename}")
    
    return df_detailed, summary_stats, comparison_df

def create_summary_statistics(preds_np, labels_np, variable_names, epoch):
    """创建汇总统计表"""
    summary_data = []
    
    for var_idx, var_name in enumerate(variable_names):
        pred_values = preds_np[:, var_idx, :].flatten()
        true_values = labels_np[:, var_idx, :].flatten()
        
        # 计算各种统计指标
        mse = np.mean((pred_values - true_values) ** 2)
        mae = np.mean(np.abs(pred_values - true_values))
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(pred_values, true_values)[0, 1] if len(pred_values) > 1 else 0
        
        # 相对误差
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
    """创建变量对比表（每个变量一列预测值，一列真实值）"""
    comparison_data = {'Epoch': epoch, 'Sample_Index': [], 'Spatial_Point': []}
    
    # 初始化列
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
    """保存所有epoch的对比结果"""
    output_path = Path(output_dir)
    comparison_filename = output_path / "all_epochs_comparison.xlsx"
    
    # 合并所有epoch的统计数据
    combined_stats = pd.concat(all_epoch_stats, ignore_index=True)
    
    # 创建透视表，便于比较不同epoch
    pivot_table = combined_stats.pivot_table(
        index='Variable', 
        columns='Epoch', 
        values=['MSE', 'MAE', 'Correlation'], 
        aggfunc='first'
    )
    
    # 保存到Excel的多个sheet
    with pd.ExcelWriter(comparison_filename) as writer:
        combined_stats.to_excel(writer, sheet_name='All_Stats', index=False)
        pivot_table.to_excel(writer, sheet_name='Epoch_Comparison')
    
    print(f"所有epoch对比结果已保存: {comparison_filename}")

def main(model, X_batch, y_batch, device):
    print("starting training...")
    
    # Create directory for saving model weights and numpy arrays
    weights_dir = Path("epoch_weights_new")
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
    all_epoch_stats = []  # 存储所有epoch的统计数据

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
            
            # 保存到Excel
            detailed_df, summary_stats, comparison_df = save_predictions_to_excel(
                all_preds, all_labels, X_batch, y_batch, epoch + 1
            )
            all_epoch_stats.append(summary_stats)

    # 保存所有epoch的对比
    if all_epoch_stats:
        save_epoch_comparison(all_epoch_stats)
    
    print("所有epoch的Excel导出完成！")
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
    mse = F.mse_loss(all_preds, all_labels)

    # Create comprehensive visualization for all variables
    create_comprehensive_visualization(all_preds, all_labels)

    # Save final evaluation results
    np.save('comprehensive_preds.npy', all_preds.cpu().numpy())
    np.save('comprehensive_labels.npy', all_labels.cpu().numpy())
    
    print(f'Mean Squared Error: {mse.item()}')


def create_comprehensive_visualization(all_preds, all_labels):
    """
    Create a comprehensive visualization showing all pollutants and variables across all 144 points
    """
    # Variable names and their indices in the tensor
    variables = {
        '2t (Temperature)': 0,
        'PM10': 1, 
        'PM2.5': 2,
        'SO2': 3,
        'NO2': 4,
        'O3': 5,
        'q (Humidity)': 6
    }
    
    # Set up the plot with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Predictions vs True Values for All Variables (All 144 Points)', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Colors for predicted and true values
    pred_color = '#FF6B6B'  # Red for predictions
    true_color = '#4ECDC4'  # Teal for true values
    
    for idx, (var_name, var_idx) in enumerate(variables.items()):
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            
            # Extract values for this variable across all samples and all 144 points
            pred_values_all_points = []
            true_values_all_points = []
            
            for i in range(len(all_preds)):
                # Extract all 144 points for this variable and sample
                pred_vals = all_preds[i][0, var_idx].cpu().numpy().flatten()  # Shape: [144]
                true_vals = all_labels[i][0, var_idx].cpu().numpy().flatten()  # Shape: [144]
                
                pred_values_all_points.extend(pred_vals)
                true_values_all_points.extend(true_vals)
            
            # Plot the lines
            point_indices = range(len(pred_values_all_points))
            ax.plot(point_indices, pred_values_all_points, color=pred_color, linewidth=1, 
                   alpha=0.7, label='Predicted')
            ax.plot(point_indices, true_values_all_points, color=true_color, linewidth=1, 
                   alpha=0.7, label='True')
            
            # Styling
            ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Point Index (All Samples × 144 Points)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add correlation coefficient
            correlation = np.corrcoef(pred_values_all_points, true_values_all_points)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplots
    for idx in range(len(variables), len(axes_flat)):
        axes_flat[idx].remove()
    
    plt.tight_layout()
    plt.savefig('comprehensive_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a single combined plot showing all variables across all points
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
    plt.title('All Variables: Predictions vs True Values (All 144 Points)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_variables_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive visualizations saved (all 144 points):")
    print("- comprehensive_predictions_vs_true.png (subplot layout)")
    print("- all_variables_combined.png (single plot with all lines)")


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
