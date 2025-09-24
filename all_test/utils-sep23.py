import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from aurora import Batch, Metadata

class WeatherDataset(Dataset):
    def __init__(self, df, sequence_length=2):
        self.df = df
        self.stations = df['station_name'].unique()
        self.sequence_length = sequence_length
        self.features = ['pm10', 'so2', 'no2', 'pm25', 'o3', 
                            'temperature', 'humidity']

        self.data = []
        for station in self.stations:
            station_data = self.df[self.df['station_name'] == station][self.features].values
            self.data.append(station_data)
        
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

def create_dataloaders(df, sequence_length, batch_size, test_size=0.2):
    dataset = WeatherDataset(df, sequence_length)
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, shuffle=False)

    train_indices = train_indices[:-(len(train_indices) % batch_size)]
    val_indices = val_indices[:-(len(val_indices) % batch_size)]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def calculate_specific_humidity(temp_k, rh_percent, pressure_pa):
    """
    Calculate specific humidity from temperature, relative humidity, and pressure
    使用Aurora标准的比湿度计算
    """
    # Tetens formula for saturation vapor pressure (Pa)
    temp_c = temp_k - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    
    # Actual vapor pressure
    e = (rh_percent / 100.0) * es
    
    # Specific humidity (kg/kg)
    q = 0.622 * e / (pressure_pa - 0.378 * e)
    
    return q

def wind_speed_dir_to_uv(wind_speed, wind_dir):
    """
    Convert wind speed and direction to u/v components for Aurora
    wind_dir: degrees (meteorological convention: direction wind comes from)
    returns: u, v components (m/s)
    """
    wind_dir_rad = np.radians(wind_dir)
    u = -wind_speed * np.sin(wind_dir_rad)  # eastward component
    v = -wind_speed * np.cos(wind_dir_rad)  # northward component
    return u, v

def uv_to_wind_speed_dir(u, v):
    """
    Convert u/v components back to wind speed and direction
    """
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = np.degrees(np.arctan2(-u, -v)) % 360
    return wind_speed, wind_dir

def specific_humidity_to_rh(q, temp_k, pressure_pa):
    """
    Convert specific humidity back to relative humidity percentage
    """
    temp_c = temp_k - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    
    # Calculate vapor pressure from specific humidity
    e = q * pressure_pa / (0.622 + 0.378 * q)
    
    # Relative humidity
    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)

def process_excel2(filename_weather: str, filename_airquality: str) -> pd.DataFrame:
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.abspath(os.path.join(base_path, filename_weather))
    airquality_file_path = os.path.abspath(os.path.join(base_path, filename_airquality))

    print(f"Weather file path: {weather_file_path}")
    print(f"Airquality file path: {airquality_file_path}")

    # 读取气象数据
    df_weather = pd.read_excel(io=weather_file_path, header=0)
    
    # 重命名列
    df_weather = df_weather[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度', 
                           '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed', 'wind_direction']
    
    # 读取空气质量数据 - 修复时间列重复问题
    df_airquality_dict = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0)
    
    # 处理每个sheet，确保列名一致
    air_data_list = []
    for sheet_name, sheet_df in df_airquality_dict.items():
        sheet_copy = sheet_df.copy()
        
        # 选择需要的列，确保列名一致
        required_columns = ['station_name', 'monitoring_time', 'longitude', 'latitude', 
                          'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        
        # 检查是否所有必要列都存在
        available_columns = []
        for col in required_columns:
            if col in sheet_copy.columns:
                available_columns.append(col)
            else:
                # 尝试查找相似的列名
                if col == 'monitoring_time':
                    time_cols = [c for c in sheet_copy.columns if 'time' in c.lower() or '时间' in c]
                    if time_cols:
                        sheet_copy.rename(columns={time_cols[0]: 'monitoring_time'}, inplace=True)
                        available_columns.append('monitoring_time')
                        print(f"{sheet_name}: 使用 {time_cols[0]} 作为时间列")
                    else:
                        print(f"警告: {sheet_name} 中找不到时间列")
                        continue
                elif col in ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']:
                    # 对于污染物数据，如果缺失就跳过
                    print(f"警告: {sheet_name} 中缺少 {col} 列")
        
        # 只选择可用的列
        if 'station_name' in available_columns and 'monitoring_time' in available_columns:
            sheet_filtered = sheet_copy[available_columns].copy()
            air_data_list.append(sheet_filtered)
        else:
            print(f"跳过sheet {sheet_name}: 缺少必要的站点名或时间列")
    
    if not air_data_list:
        print("错误: 没有有效的空气质量数据sheet")
        return pd.DataFrame()
    
    # 合并所有sheet
    df_airquality = pd.concat(air_data_list, ignore_index=True)
    df_airquality.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    print(f"空气质量数据合并后形状: {df_airquality.shape}")
    print(f"空气质量数据站点: {df_airquality['station_name'].unique()}")
    
    # 处理时间格式
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_airquality['time'] = pd.to_datetime(df_airquality['time'])

    df_weather.sort_values(by='time', inplace=True)
    df_airquality.sort_values(by='time', inplace=True)

    # 合并数据 - 使用inner join确保时间匹配
    print(f"\n合并前:")
    print(f"气象数据: {df_weather.shape}, 站点: {df_weather['station_name'].unique()}")
    print(f"空气质量数据: {df_airquality.shape}, 站点: {df_airquality['station_name'].unique()}")
    
    df_merged = pd.merge(df_weather, df_airquality, on=['station_name', 'time'], how='inner')
    
    print(f"合并后数据形状: {df_merged.shape}")
    print(f"合并后站点: {df_merged['station_name'].unique()}")
    
    if df_merged.empty:
        print("警告: 合并后数据为空，可能是时间不匹配或站点名不一致")
        return df_merged
    
    # 数据预处理
    df_merged['specific_humidity'] = calculate_specific_humidity(
        df_merged['temperature'], 
        df_merged['humidity'], 
        df_merged['pressure']
    )
    
    u_wind, v_wind = wind_speed_dir_to_uv(df_merged['wind_speed'], df_merged['wind_direction'])
    df_merged['u_wind'] = u_wind
    df_merged['v_wind'] = v_wind
    
    df_merged['pressure_hpa'] = df_merged['pressure'] / 100.0
    
    # 过滤时间范围
    df_merged = df_merged.sort_values(by='time')
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df_merged = df_merged[df_merged['time'] <= cutoff_date]
    df_merged = df_merged.reset_index(drop=True)

    # 清理数据
    df_merged.dropna(inplace=True)
    
    # 打印数据范围检查
    print("\n数据范围检查:")
    print(f"温度: {df_merged['temperature'].min():.2f} - {df_merged['temperature'].max():.2f} K")
    print(f"原始湿度: {df_merged['humidity'].min():.2f} - {df_merged['humidity'].max():.2f} %")
    print(f"比湿度: {df_merged['specific_humidity'].min():.6f} - {df_merged['specific_humidity'].max():.6f} kg/kg")
    print(f"风速: {df_merged['wind_speed'].min():.2f} - {df_merged['wind_speed'].max():.2f} m/s")
    print(f"降雨: {df_merged['rainfall'].min():.2f} - {df_merged['rainfall'].max():.2f} mm")

    return df_merged

def form_aurora_batch(df: pd.DataFrame):
    """
    Enhanced version with meteorological variables - 自适应站点数量
    """
    # 先检查有哪些站点
    stations = df['station_name'].unique()
    print(f"所有站点: {stations}")
    
    # 指定想要的4个站点：天生、龙井湾、唐家沱、上清寺（排除鱼新街）
    desired_stations = ['天生', '龙井湾', '唐家沱', '上清寺']
    
    # 检查哪些期望的站点在数据中存在
    available_stations = []
    for station in desired_stations:
        if station in stations:
            available_stations.append(station)
            station_count = (df['station_name'] == station).sum()
            print(f"找到站点 {station}: {station_count} 条记录")
        else:
            print(f"警告: 站点 {station} 不在数据中")
    
    # 如果期望的站点不足4个，补充其他站点（但排除鱼新街）
    if len(available_stations) < 4:
        print(f"期望的站点不足4个，当前可用: {available_stations}")
        # 从所有站点中选择，排除鱼新街和已选择的站点
        remaining_stations = [s for s in stations if s not in available_stations and s != '鱼新街']
        
        # 补充到4个站点
        needed = 4 - len(available_stations)
        available_stations.extend(remaining_stations[:needed])
        print(f"补充站点，最终选择: {available_stations}")
    
    # 确保选择4个站点，优先包含上清寺
    if len(available_stations) > 4:
        # 如果超过4个，优先保留上清寺
        if '上清寺' in available_stations:
            available_stations = ['天生', '龙井湾', '唐家沱', '上清寺']
        else:
            available_stations = available_stations[:4]
    
    selected_stations = available_stations
    print(f"最终选择的{len(selected_stations)}个站点: {selected_stations}")
    
    # 打印站点选择摘要
    if '上清寺' in selected_stations:
        print("✓ 上清寺已包含在分析中")
    else:
        print("✗ 上清寺未包含在分析中")
    
    if '鱼新街' in selected_stations:
        print("包含鱼新街")
    else:
        print("排除了鱼新街")
    
    # 过滤数据只包含选择的站点
    df_filtered = df[df['station_name'].isin(selected_stations)].copy()
    print(f"过滤后数据形状: {df_filtered.shape}")
    
    # 检查每个选中站点的数据量
    for station in selected_stations:
        station_count = (df_filtered['station_name'] == station).sum()
        if station_count > 0:
            time_range = df_filtered[df_filtered['station_name'] == station]['time'].agg(['min', 'max'])
            print(f"{station}: {station_count} 条记录, 时间: {time_range['min']} 到 {time_range['max']}")
        else:
            print(f"警告: {station} 没有数据！")
    
    df_sorted = df_filtered.sort_values(by=['time', 'station_name'])

    # 扩展特征列表包含所有气象变量
    features = ['temperature', 'specific_humidity', 'rainfall', 'pressure_hpa', 'u_wind', 'v_wind',
                'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    df_sorted = df_sorted.dropna()
    
    # 数据透视
    df_pivoted = df_sorted.pivot_table(index=['time'], 
                                       columns=['station_name'], 
                                       values=features)
    
    unique_times = df_pivoted.index.unique()
    t = len(unique_times)
    
    print(f"透视后数据形状: {df_pivoted.shape}")
    print(f"时间点数量: {t}")

    # 检查透视后的站点顺序
    station_order = df_pivoted.columns.get_level_values(1).unique()
    actual_station_count = len(station_order)
    print(f"透视后的站点顺序: {list(station_order)}")
    print(f"实际站点数量: {actual_station_count}")

    # 动态计算网格大小 - 根据实际站点数量
    if actual_station_count == 4:
        grid_h, grid_w = 2, 2
    elif actual_station_count == 3:
        grid_h, grid_w = 2, 2  # 3个站点也用2x2，第4个位置填充
    elif actual_station_count == 2:
        grid_h, grid_w = 1, 2
    elif actual_station_count == 1:
        grid_h, grid_w = 1, 1
    else:
        raise ValueError(f"不支持的站点数量: {actual_station_count}")
    
    print(f"使用网格大小: {grid_h}x{grid_w}")

    # 重新整形数据数组
    data_array = df_pivoted.values.reshape(t, -1, grid_h, grid_w)
    print(f"数据数组形状: {data_array.shape}")

    X_batches = []
    y_batches = []

    for i in range(1, t-2):
        if np.isnan(data_array[i-1:i+3]).any():
            continue

        # 调试特征顺序 - 只在第一次循环时打印
        if i == 1:
            feature_order = df_pivoted.columns.get_level_values(0).unique()
            print(f"实际特征顺序: {feature_order.tolist()}")
        
        # 根据实际特征顺序构建feat_map
        feat_map = {
            'co': 0, 'no2': 1, 'o3': 2, 'pm10': 3, 'pm25': 4, 'pressure_hpa': 5,
            'rainfall': 6, 'so2': 7, 'specific_humidity': 8, 'temperature': 9, 'u_wind': 10, 'v_wind': 11
        }

        # 扩展到12x12网格（Aurora标准）
        expand_h, expand_w = 12 // grid_h, 12 // grid_w
        
        X_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array[i-1:i+1, feat_map['temperature']][None]).repeat(1, 1, expand_h, expand_w),
                "10u": torch.from_numpy(data_array[i-1:i+1, feat_map['u_wind']][None]).repeat(1, 1, expand_h, expand_w),
                "10v": torch.from_numpy(data_array[i-1:i+1, feat_map['v_wind']][None]).repeat(1, 1, expand_h, expand_w),
                "msl": torch.from_numpy(data_array[i-1:i+1, feat_map['pressure_hpa']][None]).repeat(1, 1, expand_h, expand_w) * 100,
                "pm10": torch.from_numpy(data_array[i-1:i+1, feat_map['pm10']][None]).repeat(1, 1, expand_h, expand_w), 
                "pm25": torch.from_numpy(data_array[i-1:i+1, feat_map['pm25']][None]).repeat(1, 1, expand_h, expand_w), 
                "so2": torch.from_numpy(data_array[i-1:i+1, feat_map['so2']][None]).repeat(1, 1, expand_h, expand_w),
                "no2": torch.from_numpy(data_array[i-1:i+1, feat_map['no2']][None]).repeat(1, 1, expand_h, expand_w), 
                "o3": torch.from_numpy(data_array[i-1:i+1, feat_map['o3']][None]).repeat(1, 1, expand_h, expand_w),
                "co": torch.from_numpy(data_array[i-1:i+1, feat_map['co']][None]).repeat(1, 1, expand_h, expand_w),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i-1:i+1, feat_map['temperature']][None]).view((1, 2, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "q": torch.from_numpy(data_array[i-1:i+1, feat_map['specific_humidity']][None]).view((1, 2, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "u": torch.from_numpy(data_array[i-1:i+1, feat_map['u_wind']][None]).view((1, 2, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "v": torch.from_numpy(data_array[i-1:i+1, feat_map['v_wind']][None]).view((1, 2, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(unique_times[i],),
                atmos_levels=(1000,)
            ),
        )

        y_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array[i+2, feat_map['temperature']][None]).repeat(1, 1, expand_h, expand_w),
                "10u": torch.from_numpy(data_array[i+2, feat_map['u_wind']][None]).repeat(1, 1, expand_h, expand_w),
                "10v": torch.from_numpy(data_array[i+2, feat_map['v_wind']][None]).repeat(1, 1, expand_h, expand_w),
                "msl": torch.from_numpy(data_array[i+2, feat_map['pressure_hpa']][None]).repeat(1, 1, expand_h, expand_w) * 100,
                "pm10": torch.from_numpy(data_array[i+2, feat_map['pm10']][None]).repeat(1, 1, expand_h, expand_w), 
                "pm25": torch.from_numpy(data_array[i+2, feat_map['pm25']][None]).repeat(1, 1, expand_h, expand_w), 
                "so2": torch.from_numpy(data_array[i+2, feat_map['so2']][None]).repeat(1, 1, expand_h, expand_w),
                "no2": torch.from_numpy(data_array[i+2, feat_map['no2']][None]).repeat(1, 1, expand_h, expand_w), 
                "o3": torch.from_numpy(data_array[i+2, feat_map['o3']][None]).repeat(1, 1, expand_h, expand_w),
                "co": torch.from_numpy(data_array[i+2, feat_map['co']][None]).repeat(1, 1, expand_h, expand_w),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i+2, feat_map['temperature']][None]).view((1, 1, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "q": torch.from_numpy(data_array[i+2, feat_map['specific_humidity']][None]).view((1, 1, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "u": torch.from_numpy(data_array[i+2, feat_map['u_wind']][None]).view((1, 1, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
                "v": torch.from_numpy(data_array[i+2, feat_map['v_wind']][None]).view((1, 1, 1, grid_h, grid_w)).repeat(1, 1, 1, expand_h, expand_w),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(unique_times[i+2],),
                atmos_levels=(1000,)
            ),
        )

        X_batches.append(X_batch)
        y_batches.append(y_batch)

    print(f"生成了 {len(X_batches)} 个批次")
    return X_batches, y_batches

def plot_predictions_vs_labels_enhanced(all_preds, all_labels, output_prefix='predictions_vs_labels'):
    """
    Enhanced plotting function for all variables including meteorological ones (fixed version)
    """
    # 空气质量变量和气象变量 - 移除降雨量
    air_quality_vars = {
        'Temperature (K)': 0,
        'PM10 (μg/m³)': 1, 
        'PM2.5 (μg/m³)': 2,
        'SO2 (μg/m³)': 3,
        'NO2 (μg/m³)': 4,
        'O3 (μg/m³)': 5,
        'Humidity (kg/kg)': 6,
        'U Wind (m/s)': 7,
        'V Wind (m/s)': 8,
        'Pressure (Pa)': 9,
        'CO (mg/m³)': 10
    }
    
    # 创建空气质量图 - 3x4布局去掉降雨量
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Air Quality & Meteorological Predictions vs True Values (Fixed)', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, (var_name, var_idx) in enumerate(air_quality_vars.items()):
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            
            pred_values = []
            true_values = []
            
            for i in range(len(all_preds)):
                # 处理不同的数组维度
                if all_preds[i].ndim == 3:
                    pred_vals = all_preds[i][0, var_idx].cpu().numpy().flatten()
                    true_vals = all_labels[i][0, var_idx].cpu().numpy().flatten()
                else:
                    pred_vals = all_preds[i][var_idx].cpu().numpy().flatten()
                    true_vals = all_labels[i][var_idx].cpu().numpy().flatten()
                
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
            
            if len(pred_values) > 1 and len(true_values) > 1:
                correlation = np.corrcoef(pred_values, true_values)[0, 1]
                ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 移除多余的子图
    for idx in range(len(air_quality_vars), len(axes_flat)):
        axes_flat[idx].remove()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced visualization saved: {output_prefix}_enhanced.png")

def data_segmentation(df: pd.DataFrame, batch_size=16):
    """Legacy function for compatibility"""
    columns = ['pm10', 'so2', 'no2', 'pm25', 'o3',
                'temperature', 'pressure', 'humidity']
    data = df[columns].values

    X, y = [], []

    for i in range(len(data) - 2):
        X.append(data[i:i+2])
        y.append(data[i+2])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
