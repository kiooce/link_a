import os

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

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

    # 读取气象数据 - 使用你提供的列格式
    df_weather = pd.read_excel(io=weather_file_path, header=0)
    
    # 重命名列以匹配你的格式
    weather_columns = {
        'station_name': 'station_name',
        'time': 'time', 
        '温度 单位开尔文K--减去273.15换算为摄氏度': 'temperature',
        '湿度': 'humidity',
        '小时降雨量 mm': 'rainfall',
        '气压': 'pressure',
        '风速': 'wind_speed',
        '风向': 'wind_direction'
    }
    
    # 选择需要的列并重命名
    df_weather = df_weather[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度', 
                           '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed', 'wind_direction']
    
    # 读取空气质量数据
    df_airquality = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0)
    df_airquality = pd.concat(df_airquality.values(), ignore_index=True)
    
    # 选择空气质量相关列
    air_columns = ['station_name', 'monitoring_time', 'longitude', 'latitude', 
                  'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    df_airquality = df_airquality[air_columns].copy()
    df_airquality.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    # 处理时间格式
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_airquality['time'] = pd.to_datetime(df_airquality['time'])

    df_weather.sort_values(by='time', inplace=True)
    df_airquality.sort_values(by='time', inplace=True)

    # 合并数据
    df_merged = pd.merge(df_weather, df_airquality, on=['station_name', 'time'], how='inner')
    
    # 数据预处理
    # 1. 将湿度转换为比湿度 (Aurora标准)
    df_merged['specific_humidity'] = calculate_specific_humidity(
        df_merged['temperature'], 
        df_merged['humidity'], 
        df_merged['pressure']
    )
    
    # 2. 将风速风向转换为u/v分量
    u_wind, v_wind = wind_speed_dir_to_uv(df_merged['wind_speed'], df_merged['wind_direction'])
    df_merged['u_wind'] = u_wind
    df_merged['v_wind'] = v_wind
    
    # 3. 确保压力单位正确 (Aurora期待hPa)
    df_merged['pressure_hpa'] = df_merged['pressure'] / 100.0  # 转换为hPa
    
    # 过滤时间范围
    df_merged = df_merged.sort_values(by='time')
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df_merged = df_merged[df_merged['time'] <= cutoff_date]
    df_merged = df_merged.reset_index(drop=True)

    # 清理数据
    df_merged.dropna(inplace=True)
    
    # 打印数据范围以检查
    print("\n数据范围检查:")
    print(f"温度: {df_merged['temperature'].min():.2f} - {df_merged['temperature'].max():.2f} K")
    print(f"原始湿度: {df_merged['humidity'].min():.2f} - {df_merged['humidity'].max():.2f} %")
    print(f"比湿度: {df_merged['specific_humidity'].min():.6f} - {df_merged['specific_humidity'].max():.6f} kg/kg")
    print(f"风速: {df_merged['wind_speed'].min():.2f} - {df_merged['wind_speed'].max():.2f} m/s")
    print(f"降雨: {df_merged['rainfall'].min():.2f} - {df_merged['rainfall'].max():.2f} mm")

    return df_merged

def form_aurora_batch(df: pd.DataFrame):
    """
    Enhanced version with meteorological variables - 只使用4个站点匹配原代码
    """
    # 先检查有哪些站点
    stations = df['station_name'].unique()
    print(f"所有站点: {stations}")
    
    # 只选择前4个站点，保持与师哥原代码一致
    selected_stations = stations[:4]
    print(f"选择的4个站点: {selected_stations}")
    
    # 过滤数据只包含选择的站点
    df_filtered = df[df['station_name'].isin(selected_stations)].copy()
    print(f"过滤后数据形状: {df_filtered.shape}")
    
    df_sorted = df_filtered.sort_values(by=['time', 'station_name'])

    # 扩展特征列表包含所有气象变量
    features = ['temperature', 'specific_humidity', 'rainfall', 'pressure_hpa', 'u_wind', 'v_wind',
                'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    df_sorted = df_sorted.dropna()
    
    # 简化数据透视 - 不使用pressure_hpa作为索引，因为所有记录的pressure都是相同的
    df_pivoted = df_sorted.pivot_table(index=['time'], 
                                       columns=['station_name'], 
                                       values=features)
    
    unique_times = df_pivoted.index.unique()
    t = len(unique_times)
    
    print(f"透视后数据形状: {df_pivoted.shape}")
    print(f"时间点数量: {t}")

    # 重新整形数据数组 - 现在有12个特征，4个站点（2x2）
    data_array = df_pivoted.values.reshape(t, -1, 2, 2)  # t, 12 features, 2x2 stations
    
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
        
        # 根据实际打印的特征顺序构建feat_map
        # 实际顺序: ['co', 'no2', 'o3', 'pm10', 'pm25', 'pressure_hpa', 'rainfall', 'so2', 'specific_humidity', 'temperature', 'u_wind', 'v_wind']
        feat_map = {
            'co': 0, 'no2': 1, 'o3': 2, 'pm10': 3, 'pm25': 4, 'pressure_hpa': 5,
            'rainfall': 6, 'so2': 7, 'specific_humidity': 8, 'temperature': 9, 'u_wind': 10, 'v_wind': 11
        }

        X_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array[i-1:i+1, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array[i-1:i+1, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array[i-1:i+1, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array[i-1:i+1, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,  # hPa转Pa
                "pm10": torch.from_numpy(data_array[i-1:i+1, feat_map['pm10']][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array[i-1:i+1, feat_map['pm25']][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array[i-1:i+1, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array[i-1:i+1, feat_map['no2']][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array[i-1:i+1, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array[i-1:i+1, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i-1:i+1, feat_map['temperature']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array[i-1:i+1, feat_map['specific_humidity']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array[i-1:i+1, feat_map['u_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array[i-1:i+1, feat_map['v_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
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
                "2t": torch.from_numpy(data_array[i+2, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array[i+2, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array[i+2, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array[i+2, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,
                "pm10": torch.from_numpy(data_array[i+2, feat_map['pm10']][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array[i+2, feat_map['pm25']][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array[i+2, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array[i+2, feat_map['no2']][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array[i+2, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array[i+2, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i+2, feat_map['temperature']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array[i+2, feat_map['specific_humidity']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array[i+2, feat_map['u_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array[i+2, feat_map['v_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
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
