#utils
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
    从temperature，relative humidity，和pressure算specific humidity
    使用Aurora标准的比湿度计算
    """
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
    
    # 换算压力
    e = q * pressure_pa / (0.622 + 0.378 * q)
    
    # 换算relative湿度
    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)

def process_excel2(filename_weather: str, filename_airquality: str) -> pd.DataFrame:
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.abspath(os.path.join(base_path, filename_weather))
    airquality_file_path = os.path.abspath(os.path.join(base_path, filename_airquality))

    print(f"Weather file path: {weather_file_path}")
    print(f"Airquality file path: {airquality_file_path}")

    # import气象数据
    df_weather = pd.read_excel(io=weather_file_path, header=0)
    
    # weather_columns = {
    #     'station_name': 'station_name',
    #     'time': 'time', 
    #     '温度 单位开尔文K--减去273.15换算为摄氏度': 'temperature',
    #     '湿度': 'humidity',
    #     '小时降雨量 mm': 'rainfall',
    #     '气压': 'pressure',
    #     '风速': 'wind_speed',
    #     '风向': 'wind_direction'
    # }
    
    # 选择需要的列并重命名，不然看不懂
    df_weather = df_weather[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度', 
                           '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed', 'wind_direction']
    
    # import空气质量excel
    df_airquality = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0)
    df_airquality = pd.concat(df_airquality.values(), ignore_index=True)
    
    # 选择空气质量相关的列
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
    # 1. 将湿度转换为比湿度 (Aurora要求的
    df_merged['specific_humidity'] = calculate_specific_humidity(
        df_merged['temperature'], 
        df_merged['humidity'], 
        df_merged['pressure']
    )
    
    # 2. 将风速风向转换为u/v分量（Aurora要求
    u_wind, v_wind = wind_speed_dir_to_uv(df_merged['wind_speed'], df_merged['wind_direction'])
    df_merged['u_wind'] = u_wind
    df_merged['v_wind'] = v_wind
    
    # 3. 确保压力单位正确 (Aurora要hPa，所以得做换算？
    df_merged['pressure_hpa'] = df_merged['pressure'] / 100.0  # 转换为hPa
    
    # 过滤时间
    df_merged = df_merged.sort_values(by='time')
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df_merged = df_merged[df_merged['time'] <= cutoff_date]
    df_merged = df_merged.reset_index(drop=True)

    # 检查每个站点的NaN
    print("\nNaN情况")
    for station in df_merged['station_name'].unique():
        station_data = df_merged[df_merged['station_name'] == station]
        nan_count = station_data.isna().sum().sum()
        total_cells = len(station_data) * len(station_data.columns)
        print(f"  {station}: {nan_count}/{total_cells} 个NaN值")
        
        # 找出NaN，也就是空白数据
        nan_columns = station_data.columns[station_data.isna().any()].tolist()
        if nan_columns:
            print(f"    NaN列: {nan_columns}")

    cleaned_data_list = []
    for station in df_merged['station_name'].unique():
        station_data = df_merged[df_merged['station_name'] == station].copy()
        before_count = len(station_data)
        required_columns = ['temperature', 'pressure', 'pm25', 'pm10']
        # 删除部分NaN
        station_data = station_data.dropna(subset=required_columns)
        after_count = len(station_data)
        
        cleaned_data_list.append(station_data)
        print(f"  {station}: {before_count} -> {after_count} 条记录 (删除了 {before_count-after_count} 条)")

    df_merged = pd.concat(cleaned_data_list, ignore_index=True)

    print("清理后还剩的数据")
    for station in df_merged['station_name'].unique():
        count = len(df_merged[df_merged['station_name'] == station])
        print(f"{station}: {count} 条")
    
    # 打印数据范围以检查
    print("\n范围check:")
    print(f"温度: {df_merged['temperature'].min():.2f} - {df_merged['temperature'].max():.2f} K")
    print(f"原始湿度: {df_merged['humidity'].min():.2f} - {df_merged['humidity'].max():.2f} %")
    print(f"比湿度: {df_merged['specific_humidity'].min():.6f} - {df_merged['specific_humidity'].max():.6f} kg/kg")
    print(f"风速: {df_merged['wind_speed'].min():.2f} - {df_merged['wind_speed'].max():.2f} m/s")
    print(f"降雨: {df_merged['rainfall'].min():.2f} - {df_merged['rainfall'].max():.2f} mm")

    return df_merged

def form_aurora_batch(df: pd.DataFrame):
    """
    修复版本：确保站点映射正确
    """
    stations = df['station_name'].unique()
    print(f"所有站点: {stations}")
    
    # 严格按顺序选择站点
    desired_stations = ['上清寺', '唐家沱', '天生', '龙井湾']
    available_stations = []

    print("=== 严格验证站点可用性 ===")
    for station in desired_stations:
        if station in stations:
            station_data = df[df['station_name'] == station]
            data_count = len(station_data)
            available_stations.append(station)
            print(f"✓ {station}: {data_count}条数据")
        else:
            print(f"✗ {station}: 不可用")
            
    if len(available_stations) != 4:
        raise ValueError(f"必须有4个站点，当前只有{len(available_stations)}个: {available_stations}")

    # 只使用严格顺序的4个站点
    selected_stations = available_stations
    print(f"最终选择: {selected_stations}")
    
    # 过滤数据
    df_filtered = df[df['station_name'].isin(selected_stations)].copy()
    print(f"过滤后数据: {df_filtered.shape}")
    
    # 强制排序：先按时间，再按站点（确保站点顺序）
    df_filtered['station_order'] = df_filtered['station_name'].map({
        '上清寺': 0, '唐家沱': 1, '天生': 2, '龙井湾': 3
    })
    df_sorted = df_filtered.sort_values(by=['time', 'station_order'])
    
    # 删除辅助列
    df_sorted = df_sorted.drop('station_order', axis=1)
    
    features = ['temperature', 'specific_humidity', 'rainfall', 'pressure_hpa', 'u_wind', 'v_wind',
                'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    # 清理NaN
    df_sorted = df_sorted.dropna()
    
    # === 关键修复：手动构建透视表确保顺序 ===
    print("=== 手动构建透视表确保站点顺序 ===")
    
    unique_times = df_sorted['time'].unique()
    unique_times = sorted(unique_times)  # 确保时间排序
    
    # 预分配数组
    n_times = len(unique_times)
    n_features = len(features)
    n_stations = 4
    
    # 创建 [time, feature, station] 的3D数组
    data_array_3d = np.full((n_times, n_features, n_stations), np.nan)
    
    # 逐时间点填充数据
    valid_times = []
    
    for t_idx, time_point in enumerate(unique_times):
        time_data = df_sorted[df_sorted['time'] == time_point]
        
        # 检查是否所有4个站点都有数据
        stations_in_time = time_data['station_name'].unique()
        if len(stations_in_time) != 4:
            continue  # 跳过不完整的时间点
            
        # 按严格顺序填充站点数据
        all_stations_present = True
        
        for s_idx, station in enumerate(selected_stations):
            station_data = time_data[time_data['station_name'] == station]
            
            if len(station_data) != 1:
                all_stations_present = False
                break
                
            # 填充特征数据
            for f_idx, feature in enumerate(features):
                if feature in station_data.columns:
                    value = station_data[feature].iloc[0]
                    if not np.isnan(value):
                        data_array_3d[t_idx, f_idx, s_idx] = value
                    else:
                        all_stations_present = False
                        break
            
            if not all_stations_present:
                break
        
        if all_stations_present:
            valid_times.append(time_point)
        
    # 只保留完整的时间点
    valid_indices = [i for i, t in enumerate(unique_times) if t in valid_times]
    data_array_3d = data_array_3d[valid_indices]
    unique_times = np.array(valid_times)
    
    print(f"完整数据时间点: {len(valid_times)} / {len(unique_times)}")
    
    # 验证数据映射正确性
    print("=== 验证数据映射 ===")
    test_time_idx = 0
    if len(valid_times) > 0:
        test_time = valid_times[test_time_idx]
        temp_feature_idx = features.index('temperature')
        
        print(f"测试时间: {test_time}")
        print("构建的数组中各站点温度:")
        for s_idx, station in enumerate(selected_stations):
            temp_val = data_array_3d[test_time_idx, temp_feature_idx, s_idx]
            print(f"  {station} [位置{s_idx}]: {temp_val:.2f}K")
        
        # 与原始数据对比
        print("原始数据中对应温度:")
        test_original = df_sorted[df_sorted['time'] == test_time]
        for _, row in test_original.iterrows():
            print(f"  {row['station_name']}: {row['temperature']:.2f}K")
    
    # 重塑为2x2网格 (t, features, 2, 2)
    data_array_4d = data_array_3d.reshape(len(valid_times), n_features, 2, 2)
    
    print(f"最终数据形状: {data_array_4d.shape}")
    print("2x2映射:")
    print("  (0,0): 上清寺")
    print("  (0,1): 唐家沱") 
    print("  (1,0): 天生")
    print("  (1,1): 龙井湾")
    
    # 生成Aurora批次
    X_batches = []
    y_batches = []
    
    # 特征映射（基于实际数据中的特征顺序）
    feat_map = {}
    for idx, feature in enumerate(features):
        feat_map[feature] = idx
    
    print(f"特征映射: {feat_map}")
    
    for i in range(1, len(valid_times)-2):
        if np.isnan(data_array_4d[i-1:i+3]).any():
            continue
            
        X_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,
                "pm10": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pm10']][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pm25']][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['no2']][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['temperature']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['specific_humidity']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['u_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['v_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(valid_times[i],),
                atmos_levels=(1000,)
            ),
        )

        y_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array_4d[i+2, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array_4d[i+2, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array_4d[i+2, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array_4d[i+2, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,
                "pm10": torch.from_numpy(data_array_4d[i+2, feat_map['pm10']][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array_4d[i+2, feat_map['pm25']][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array_4d[i+2, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array_4d[i+2, feat_map['no2']][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array_4d[i+2, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array_4d[i+2, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array_4d[i+2, feat_map['temperature']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array_4d[i+2, feat_map['specific_humidity']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array_4d[i+2, feat_map['u_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array_4d[i+2, feat_map['v_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(valid_times[i+2],),
                atmos_levels=(1000,)
            ),
        )

        X_batches.append(X_batch)
        y_batches.append(y_batch)

    print(f"生成批次数: {len(X_batches)}")
    
    # 再次验证第一个批次
    if len(X_batches) > 0:
        print("=== 验证气压 ===")
        test_X = X_batches[0]
        input_2x2 = test_X.surf_vars['msl'][0, 1]
        input_time = test_X.metadata.time[0]
        
        print(f"批次输入时间: {input_time}")
        print("批次2x2温度:")
        print(f"  (0,0): {input_2x2[0,0]:.2f}pa - 上清寺")
        print(f"  (0,1): {input_2x2[0,1]:.2f}pa - 唐家沱")
        print(f"  (1,0): {input_2x2[1,0]:.2f}pa - 天生") 
        print(f"  (1,1): {input_2x2[1,1]:.2f}pa - 龙井湾")
        
        # 与原始数据最终对比
        original_at_time = df_sorted[df_sorted['time'] == input_time]
        print("原始数据对应气压:")
        station_map = {}
        for _, row in original_at_time.iterrows():
            station_map[row['station_name']] = row['pressure']
            print(f"  {row['station_name']}: {row['pressure']:.2f}Pa")
        
        # 验证映射
        mapping_correct = True
        expected_mapping = {
            '上清寺': (0, 0), '唐家沱': (0, 1), 
            '天生': (1, 0), '龙井湾': (1, 1)
        }
        
        for station, (row, col) in expected_mapping.items():
            if station in station_map:
                orig_temp = station_map[station]
                batch_temp = input_2x2[row, col].item()
                diff = abs(orig_temp - batch_temp)
                status = "✓" if diff < 0.01 else "✗"
                print(f"验证 {station}: 原始{orig_temp:.2f} vs 批次{batch_temp:.2f} {status}")
                if diff >= 0.01:
                    mapping_correct = False
        
        if mapping_correct:
            print("🎉 站点映射验证通过！")
        else:
            print("❌ 站点映射仍有问题！")
    
    return X_batches, y_batches

def fix_visualization_ranges():
    """
    修复可视化和Excel输出中的站点范围
    """
    # 新的正确映射
    station_ranges_correct = {
        '上清寺': slice(0, 36),      # 对应2x2网格(0,0)区域
        '唐家沱': slice(36, 72),     # 对应2x2网格(0,1)区域
        '天生': slice(72, 108),      # 对应2x2网格(1,0)区域
        '龙井湾': slice(108, 144)    # 对应2x2网格(1,1)区域
    }
    
    station_points_correct = [0, 36, 72, 108]  # 每个区域的起始点
    
    return station_ranges_correct, station_points_correct

def plot_predictions_vs_labels_enhanced(all_preds, all_labels, output_prefix='predictions_vs_labels'):
    """
    Enhanced plotting function for all variables including meteorological ones (fixed version)
    """
    # 降雨量aurora不接受就移除了
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
    
    # 画可视图看结果
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Air Quality & Meteorological Predictions vs True Values (Fixed)', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, (var_name, var_idx) in enumerate(air_quality_vars.items()):
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            
            pred_values = []
            true_values = []
            
            for i in range(len(all_preds)):
                # 处理不同的数组维度（全是2纬，最开始以为是3纬，所以设俩juct in case
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
    
    # 删掉多余的子图
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
